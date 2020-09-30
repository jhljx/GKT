import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network.
# For more information, please refer to https://dl.acm.org/doi/10.1145/3350546.3352513
# Author: jhljx
# Email: jhljx8918@gmail.com


# Multi-Layer Perceptron(MLP) layer
class MLP(nn.Module):
    """Two-layer fully-connected ReLU net with batch norm."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0., bias=True):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.norm = nn.BatchNorm1d(output_dim)
        # the paper said they added Batch Normalization for the output of MLPs, as shown in Section 4.2
        self.dropout = dropout
        self.output_dim = output_dim
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        if inputs.numel() == self.output_dim or inputs.numel() == 0:
            # batch_size == 1 or 0 will cause BatchNorm error, so return the input directly
            return inputs
        if len(inputs.size()) == 3:
            x = inputs.view(inputs.size(0) * inputs.size(1), -1)
            x = self.norm(x)
            return x.view(inputs.size(0), inputs.size(1), -1)
        else:  # len(input_size()) == 2
            return self.norm(inputs)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.dropout(x, self.dropout, training=self.training)  # pay attention to add training=self.training
        x = F.relu(self.fc2(x))
        return self.batch_norm(x)


class EraseAddGate(nn.Module):
    """
    Erase & Add Gate module
    NOTE: this erase & add gate is a bit different from that in DKVMN.
    For more information about Erase & Add gate, please refer to the paper "Dynamic Key-Value Memory Networks for Knowledge Tracing"
    The paper can be found in https://arxiv.org/abs/1611.08108
    """

    def __init__(self, feature_dim, concept_num, bias=True):
        super(EraseAddGate, self).__init__()
        # weight
        self.weight = nn.Parameter(torch.rand(concept_num))
        self.reset_parameters()
        # erase gate
        self.erase = nn.Linear(feature_dim, feature_dim, bias=bias)
        # add gate
        self.add = nn.Linear(feature_dim, feature_dim, bias=bias)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        r"""
        Params:
            x: input feature matrix
        Shape:
            x: [batch_size, concept_num, feature_dim]
            res: [batch_size, concept_num, feature_dim]
        Return:
            res: returned feature matrix with old information erased and new information added
        The GKT paper didn't provide detailed explanation about this erase-add gate. As the erase-add gate in the GKT only has one input parameter,
        this gate is different with that of the DKVMN. We used the input matrix to build the erase and add gates, rather than $\mathbf{v}_{t}$ vector in the DKVMN.
        """
        erase_gate = torch.sigmoid(self.erase(x))  # [batch_size, concept_num, feature_dim]
        # self.weight.unsqueeze(dim=1) shape: [concept_num, 1]
        tmp_x = x - self.weight.unsqueeze(dim=1) * erase_gate * x
        add_feat = torch.tanh(self.add(x))  # [batch_size, concept_num, feature_dim]
        res = tmp_x + self.weight.unsqueeze(dim=1) * add_feat
        return res


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    NOTE: Stole and modify from https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Modules.py
    """

    def __init__(self, temperature, attn_dropout=0.):
        super().__init__()
        self.temperature = temperature
        self.dropout = attn_dropout

    def forward(self, q, k, mask=None):
        r"""
        Parameters:
            q: multi-head query matrix
            k: multi-head key matrix
            mask: mask matrix
        Shape:
            q: [n_head, mask_num, embedding_dim]
            k: [n_head, concept_num, embedding_dim]
        Return: attention score of all queries
        """
        attn = torch.matmul(q / self.temperature, k.transpose(1, 2))  # [n_head, mask_number, concept_num]
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        # pay attention to add training=self.training!
        attn = F.dropout(F.softmax(attn, dim=0), self.dropout, training=self.training)  # pay attention that dim=-1 is not as good as dim=0!
        return attn


class MLPEncoder(nn.Module):
    """
    MLP encoder module.
    NOTE: Stole and modify the code from https://github.com/ethanfetaya/NRI/blob/master/modules.py
    """
    def __init__(self, input_dim, hidden_dim, output_dim, factor=True, dropout=0., bias=True):
        super(MLPEncoder, self).__init__()
        self.factor = factor
        self.mlp = MLP(input_dim * 2, hidden_dim, hidden_dim, dropout=dropout, bias=bias)
        self.mlp2 = MLP(hidden_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias)
        if self.factor:
            self.mlp3 = MLP(hidden_dim * 3, hidden_dim, hidden_dim, dropout=dropout, bias=bias)
        else:
            self.mlp3 = MLP(hidden_dim * 2, hidden_dim, hidden_dim, dropout=dropout, bias=bias)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def node2edge(self, x, sp_send, sp_rec):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(sp_rec, x)
        senders = torch.matmul(sp_send, x)
        edges = torch.cat([senders, receivers], dim=1)
        return edges

    def edge2node(self, x, sp_send_t, sp_rec_t):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(sp_rec_t, x)
        return incoming

    def forward(self, inputs, sp_send, sp_rec, sp_send_t, sp_rec_t):
        r"""
        Parameters:
            inputs: input concept embedding matrix
            sp_send: one-hot encoded send-node index(sparse tensor)
            sp_rec: one-hot encoded receive-node index(sparse tensor)
            sp_send_t: one-hot encoded send-node index(sparse tensor, transpose)
            sp_rec_t: one-hot encoded receive-node index(sparse tensor, transpose)
        Shape:
            inputs: [concept_num, embedding_dim]
            sp_send: [edge_num, concept_num]
            sp_rec: [edge_num, concept_num]
            sp_send_t: [concept_num, edge_num]
            sp_rec_t: [concept_num, edge_num]
        Return:
            output: [edge_num, edge_type_num]
        """
        x = self.node2edge(inputs, sp_send, sp_rec)  # [edge_num, 2 * embedding_dim]
        x = self.mlp(x)  # [edge_num, hidden_num]
        x_skip = x

        if self.factor:
            x = self.edge2node(x, sp_send_t, sp_rec_t)  # [concept_num, hidden_num]
            x = self.mlp2(x)  # [concept_num, hidden_num]
            x = self.node2edge(x, sp_send, sp_rec)  # [edge_num, 2 * hidden_num]
            x = torch.cat((x, x_skip), dim=1)  # Skip connection  shape: [edge_num, 3 * hidden_num]
            x = self.mlp3(x)  # [edge_num, hidden_num]
        else:
            x = self.mlp2(x)  # [edge_num, hidden_num]
            x = torch.cat((x, x_skip), dim=1)  # Skip connection  shape: [edge_num, 2 * hidden_num]
            x = self.mlp3(x)  # [edge_num, hidden_num]
        output = self.fc_out(x)  # [edge_num, output_dim]
        return output


class MLPDecoder(nn.Module):
    """
    MLP decoder module.
    NOTE: Stole and modify the code from https://github.com/ethanfetaya/NRI/blob/master/modules.py
    """

    def __init__(self, input_dim, msg_hidden_dim, msg_output_dim, hidden_dim, edge_type_num, dropout=0., bias=True):
        super(MLPDecoder, self).__init__()
        self.msg_out_dim = msg_output_dim
        self.edge_type_num = edge_type_num
        self.dropout = dropout

        self.msg_fc1 = nn.ModuleList([nn.Linear(2 * input_dim, msg_hidden_dim, bias=bias) for _ in range(edge_type_num)])
        self.msg_fc2 = nn.ModuleList([nn.Linear(msg_hidden_dim, msg_output_dim, bias=bias) for _ in range(edge_type_num)])
        self.out_fc1 = nn.Linear(msg_output_dim, hidden_dim, bias=bias)
        self.out_fc2 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.out_fc3 = nn.Linear(hidden_dim, input_dim, bias=bias)

    def node2edge(self, x, sp_send, sp_rec):
        receivers = torch.matmul(sp_rec, x)  # [edge_num, embedding_dim]
        senders = torch.matmul(sp_send, x)  # [edge_num, embedding_dim]
        edges = torch.cat([senders, receivers], dim=-1)  # [edge_num, 2 * embedding_dim]
        return edges

    def edge2node(self, x, sp_send_t, sp_rec_t):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(sp_rec_t, x)
        return incoming

    def forward(self, inputs, rel_type, sp_send, sp_rec, sp_send_t, sp_rec_t):
        r"""
        Parameters:
            inputs: input concept embedding matrix
            rel_type: inferred edge weights for all edge types from MLPEncoder
            sp_send: one-hot encoded send-node index(sparse tensor)
            sp_rec: one-hot encoded receive-node index(sparse tensor)
            sp_send_t: one-hot encoded send-node index(sparse tensor, transpose)
            sp_rec_t: one-hot encoded receive-node index(sparse tensor, transpose)
        Shape:
            inputs: [concept_num, embedding_dim]
            sp_send: [edge_num, concept_num]
            sp_rec: [edge_num, concept_num]
            sp_send_t: [concept_num, edge_num]
            sp_rec_t: [concept_num, edge_num]
        Return:
            output: [edge_num, edge_type_num]
        """
        # NOTE: Assumes that we have the same graph across all samples.
        # Node2edge
        pre_msg = self.node2edge(inputs, sp_send, sp_rec)
        all_msgs = Variable(torch.zeros(pre_msg.size(0), self.msg_out_dim, device=inputs.device))  # [edge_num, msg_out_dim]
        for i in range(self.edge_type_num):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, self.dropout, training=self.training)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * rel_type[:, i:i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = self.edge2node(all_msgs, sp_send_t, sp_rec_t)  # [concept_num, msg_out_dim]
        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(agg_msgs)), self.dropout, training=self.training)  # [concept_num, hidden_dim]
        pred = F.dropout(F.relu(self.out_fc2(pred)), self.dropout, training=self.training)  # [concept_num, hidden_dim]
        pred = self.out_fc3(pred)  # [concept_num, embedding_dim]
        return pred