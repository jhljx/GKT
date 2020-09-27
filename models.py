# coding: utf-8
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from layers import MLP, EraseAddGate, MLPEncoder, MLPDecoder, ScaledDotProductAttention
from utils import gumbel_softmax

# Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network.
# For more information, please refer to https://dl.acm.org/doi/10.1145/3350546.3352513
# Author: jhljx
# Email: jhljx8918@gmail.com


class GKT(nn.Module):

    def __init__(self, concept_num, hidden_dim, embedding_dim, edge_type_num, graph_type, graph=None, graph_model=None, dropout=0.5, bias=True):
        super(GKT, self).__init__()
        self.concept_num = concept_num
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.edge_type_num = edge_type_num

        assert graph_type in ['Dense', 'Transition', 'DKT', 'PAM', 'MHA', 'VAE']
        self.graph_type = graph_type
        if graph_type in ['Dense', 'Transition', 'DKT']:
            assert edge_type_num == 2
            assert graph is not None
            self.graph = graph  # [concept_num, concept_num]
            self.graph_model = graph_model
        else:  # ['PAM', 'MHA', 'VAE']
            assert graph is None
            self.graph = graph  # None
            if graph_type == 'PAM':
                self.graph = nn.Parameter(torch.rand(concept_num, concept_num))
            self.graph_model = graph_model

        # one-hot feature and question
        self.one_hot_feat = torch.eye(2 * self.concept_num)
        self.one_hot_q = torch.eye(self.concept_num)
        self.one_hot_q = torch.cat((self.one_hot_q, torch.zeros(1, self.concept_num)), dim=0)
        # concept and concept & response embeddings
        self.emb_x = nn.Embedding(2 * concept_num, embedding_dim)
        # last embedding is used for padding, so dim + 1
        self.emb_c = nn.Embedding(concept_num + 1, embedding_dim, padding_idx=-1)

        # f_self function and f_neighbor functions
        mlp_input_dim = hidden_dim + embedding_dim
        self.f_self = MLP(mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias)
        self.f_neighbor_list = nn.ModuleList()
        if graph_type in ['Dense', 'Transition', 'DKT', 'PAM']:
            # f_in and f_out functions
            self.f_neighbor_list.append(MLP(2 * mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias))
            self.f_neighbor_list.append(MLP(2 * mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias))
        else:  # ['MHA', 'VAE']
            for i in range(edge_type_num):
                self.f_neighbor_list.append(MLP(2 * mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias))

        # Erase & Add Gate
        self.erase_add_gate = EraseAddGate(hidden_dim, concept_num)
        # Gate Recurrent Unit
        self.gru = nn.GRUCell(hidden_dim, hidden_dim, bias=bias)
        # prediction layer
        self.predict = nn.Linear(hidden_dim, 1, bias=bias)

    # Aggregate step, as shown in Section 3.2.1 of the paper
    def _aggregate(self, xt, qt, ht, batch_size):
        r"""
        Parameters:
            xt: input one-hot question answering features at the current timestamp
            qt: question indices for all students in a batch at the current timestamp
            ht: hidden representations of all concepts at the current timestamp
            batch_size: the size of a student batch
        Shape:
            xt: [batch_size]
            qt: [batch_size]
            ht: [batch_size, concept_num, hidden_dim]
            tmp_ht: [batch_size, concept_num, hidden_dim + embedding_dim]
        Return:
            tmp_ht: aggregation results of concept hidden knowledge state and concept(& response) embedding
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        x_idx_mat = torch.arange(2 * self.concept_num, device=xt.device)
        x_embedding = self.emb_x(x_idx_mat)  # [2 * concept_num, embedding_dim]
        masked_feat = F.embedding(xt[qt_mask], self.one_hot_feat)  # [mask_num, 2 * concept_num]
        res_embedding = masked_feat.mm(x_embedding)  # [mask_num, embedding_dim]
        mask_num = res_embedding.shape[0]

        concept_idx_mat = self.concept_num * torch.ones((batch_size, self.concept_num), device=xt.device).long()
        concept_idx_mat[qt_mask, :] = torch.arange(self.concept_num, device=xt.device)
        concept_embedding = self.emb_c(concept_idx_mat)  # [batch_size, concept_num, embedding_dim]

        index_tuple = (torch.arange(mask_num, device=xt.device), qt[qt_mask].long())
        concept_embedding[qt_mask] = concept_embedding[qt_mask].index_put(index_tuple, res_embedding)
        tmp_ht = torch.cat((ht, concept_embedding), dim=-1)  # [batch_size, concept_num, hidden_dim + embedding_dim]
        return tmp_ht

    # GNN aggregation step, as shown in 3.3.2 Equation 1 of the paper
    def _agg_neighbors(self, tmp_ht, qt):
        r"""
        Parameters:
            tmp_ht: temporal hidden representations of all concepts after the aggregate step
            qt: question indices for all students in a batch at the current timestamp
        Shape:
            tmp_ht: [batch_size, concept_num, hidden_dim + embedding_dim]
            qt: [batch_size]
            m_next: [batch_size, concept_num, hidden_dim]
        Return:
            m_next: hidden representations of all concepts aggregating neighboring representations at the next timestamp
            concept_embedding: input of VAE (optional)
            rec_embedding: reconstructed input of VAE (optional)
            z_prob: probability distribution of latent variable z in VAE (optional)
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        masked_qt = qt[qt_mask]  # [mask_num, ]
        masked_tmp_ht = tmp_ht[qt_mask]  # [mask_num, concept_num, hidden_dim + embedding_dim]
        mask_num = masked_tmp_ht.shape[0]
        self_index_tuple = (torch.arange(mask_num, device=qt.device), masked_qt.long())
        self_ht = masked_tmp_ht[self_index_tuple]  # [mask_num, hidden_dim + embedding_dim]
        self_features = self.f_self(self_ht)  # [mask_num, hidden_dim]
        expanded_self_ht = self_ht.unsqueeze(dim=1).repeat(1, self.concept_num, 1)  #[mask_num, concept_num, hidden_dim + embedding_dim]
        neigh_ht = torch.cat((expanded_self_ht, masked_tmp_ht), dim=-1)  #[mask_num, concept_num, 2 * (hidden_dim + embedding_dim)]
        concept_embedding, rec_embedding, z_prob = None, None, None

        if self.graph_type in ['Dense', 'Transition', 'DKT', 'PAM']:
            adj = self.graph[masked_qt.long(), :].unsqueeze(dim=-1)  # [mask_num, concept_num, 1]
            reverse_adj = self.graph[:, masked_qt.long()].transpose(0, 1).unsqueeze(dim=-1)  # [mask_num, concept_num, 1]
            # self.f_neighbor_list[0](neigh_ht) shape: [mask_num, concept_num, hidden_dim]
            neigh_features = adj * self.f_neighbor_list[0](neigh_ht) + reverse_adj * self.f_neighbor_list[1](neigh_ht)
        else:  # ['MHA', 'VAE']
            concept_index = torch.arange(self.concept_num, device=qt.device)
            concept_embedding = self.emb_c(concept_index)  # [concept_num, embedding_dim]
            if self.graph_type == 'MHA':
                query = self.emb_c(masked_qt)
                key = concept_embedding
                att_mask = Variable(torch.ones(self.edge_type_num, mask_num, self.concept_num, device=qt.device))
                # print('edge_type_number: ', self.edge_type_num)
                for k in range(self.edge_type_num):
                    index_tuple = (torch.arange(mask_num, device=qt.device), masked_qt.long())
                    # tmp_mask = att_mask[k]
                    #print('tmp_mask shape: ', tmp_mask.shape)
                    # print('mask_number: ', mask_num)
                    att_mask[k] = att_mask[k].index_put(index_tuple, torch.zeros(mask_num, device=qt.device))
                graphs = self.graph_model(masked_qt, query, key, att_mask)
            else:  # self.graph_type == 'VAE'
                rel_send, rel_rec = self._get_edges(masked_qt)
                graphs, rec_embedding, z_prob = self.graph_model(concept_embedding, rel_send, rel_rec)
            neigh_features = 0
            for k in range(self.edge_type_num):
                adj = graphs[k][masked_qt, :].unsqueeze(dim=-1)  # [mask_num, concept_num, 1]
                if k == 0:
                    neigh_features = adj * self.f_neighbor_list[k](neigh_ht)
                else:
                    neigh_features = neigh_features + adj * self.f_neighbor_list[k](neigh_ht)
            if self.graph_type == 'MHA':
                neigh_features = 1. / self.edge_type_num * neigh_features
        # neigh_features: [mask_num, concept_num, hidden_dim]
        m_next = tmp_ht[:, :, :self.hidden_dim]
        m_next[qt_mask] = neigh_features
        m_next[qt_mask] = m_next[qt_mask].index_put(self_index_tuple, self_features)
        return m_next, concept_embedding, rec_embedding, z_prob

    # Update step, as shown in Section 3.3.2 of the paper
    def _update(self, tmp_ht, ht, qt):
        r"""
        Parameters:
            tmp_ht: temporal hidden representations of all concepts after the aggregate step
            ht: hidden representations of all concepts at the current timestamp
            qt: question indices for all students in a batch at the current timestamp
        Shape:
            tmp_ht: [batch_size, concept_num, hidden_dim + embedding_dim]
            ht: [batch_size, concept_num, hidden_dim]
            qt: [batch_size]
            h_next: [batch_size, concept_num, hidden_dim]
        Return:
            h_next: hidden representations of all concepts at the next timestamp
            concept_embedding: input of VAE (optional)
            rec_embedding: reconstructed input of VAE (optional)
            z_prob: probability distribution of latent variable z in VAE (optional)
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        mask_num = qt_mask.nonzero().shape[0]
        # GNN Aggregation
        m_next, concept_embedding, rec_embedding, z_prob = self._agg_neighbors(tmp_ht, qt)  # [batch_size, concept_num, hidden_dim]
        # Erase & Add Gate
        m_next[qt_mask] = self.erase_add_gate(m_next[qt_mask])  # [mask_num, concept_num, hidden_dim]
        # GRU
        h_next = m_next
        res = self.gru(m_next[qt_mask].reshape(-1, self.hidden_dim), ht[qt_mask].reshape(-1, self.hidden_dim))  # [mask_num * concept_num, hidden_num]
        index_tuple = (torch.arange(mask_num, device=qt_mask.device), )
        h_next[qt_mask] = h_next[qt_mask].index_put(index_tuple, res.reshape(-1, self.concept_num, self.hidden_dim))
        return h_next, concept_embedding, rec_embedding, z_prob

    # Predict step, as shown in Section 3.3.3 of the paper
    def _predict(self, h_next, qt):
        r"""
        Parameters:
            h_next: hidden representations of all concepts at the next timestamp after the update step
            qt: question indices for all students in a batch at the current timestamp
        Shape:
            h_next: [batch_size, concept_num, hidden_dim]
            qt: [batch_size]
            y: [batch_size, concept_num]
        Return:
            y: predicted correct probability of all concepts at the next timestamp
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        y = self.predict(h_next).squeeze(dim=-1)  # [batch_size, concept_num]
        y[qt_mask] = torch.sigmoid(y[qt_mask])  # [batch_size, concept_num]
        return y

    def _get_next_pred(self, yt, q_next):
        r"""
        Parameters:
            yt: predicted correct probability of all concepts at the next timestamp
            q_next: question index matrix at the next timestamp
            batch_size: the size of a student batch
        Shape:
            y: [batch_size, concept_num]
            questions: [batch_size, seq_len]
            pred: [batch_size, ]
        Return:
            pred: predicted correct probability of the question answered at the next timestamp
        """
        next_qt = q_next
        next_qt = torch.where(next_qt != -1, next_qt, self.concept_num * torch.ones_like(next_qt, device=yt.device))
        one_hot_qt = F.embedding(next_qt.long(), self.one_hot_q)  # [batch_size, concept_num]
        # dot product between yt and one_hot_qt
        pred = (yt * one_hot_qt).sum(dim=1)  # [batch_size, ]
        return pred

    # Get edges for edge inference in VAE
    def _get_edges(self, masked_qt):
        r"""
        Parameters:
            masked_qt: qt index with -1 padding values removed
        Shape:
            masked_qt: [mask_num, ]
            rel_send: [edge_num, concept_num]
            rel_rec: [edge_num, concept_num]
        Return:
            rel_send: from nodes in edges which send messages to other nodes
            rel_rec:  to nodes in edges which receive messages from other nodes
        """
        mask_num = masked_qt.shape[0]
        row_arr = masked_qt.cpu().numpy().reshape(-1, 1)  # [mask_num, 1]
        row_arr = np.repeat(row_arr, self.concept_num, axis=1)  # [mask_num, concept_num]
        col_arr = np.arange(self.concept_num).reshape(1, -1)  # [1, concept_num]
        col_arr = np.repeat(col_arr, mask_num, axis=0)  # [mask_num, concept_num]
        # add reversed edges
        new_row = np.vstack((row_arr, col_arr))  # [2 * mask_num, concept_num]
        new_col = np.vstack((col_arr, row_arr))  # [2 * mask_num, concept_num]
        row_arr = new_row.flatten()  # [2 * mask_num * concept_num, ]
        col_arr = new_col.flatten()  # [2 * mask_num * concept_num, ]
        data_arr = np.ones(2 * mask_num * self.concept_num)
        init_graph = sp.coo_matrix((data_arr, (row_arr, col_arr)), shape=(self.concept_num, self.concept_num))
        init_graph.setdiag(0)  # remove self-loop edges
        row_arr, col_arr, _ = sp.find(init_graph)
        row_tensor = torch.from_numpy(row_arr).long()
        col_tensor = torch.from_numpy(col_arr).long()
        one_hot_table = torch.eye(self.concept_num, self.concept_num)
        rel_rec = F.embedding(row_tensor, one_hot_table)  # [edge_num, concept_num]
        rel_send = F.embedding(col_tensor, one_hot_table)  # [edge_num, concept_num]
        rel_send = rel_send.to(device=masked_qt.device)
        rel_rec = rel_rec.to(device=masked_qt.device)
        return rel_send, rel_rec

    def forward(self, features, questions):
        r"""
        Parameters:
            features: input one-hot matrix
            questions: question index matrix
        seq_len dimension needs padding, because different students may have learning sequences with different lengths.
        Shape:
            features: [batch_size, seq_len]
            questions: [batch_size, seq_len]
            pred_res: [batch_size, seq_len - 1]
        Return:
            pred_res: the correct probability of questions answered at the next timestamp
            concept_embedding: input of VAE (optional)
            rec_embedding: reconstructed input of VAE (optional)
            z_prob: probability distribution of latent variable z in VAE (optional)
        """
        batch_size, seq_len = features.shape
        ht = Variable(torch.zeros((batch_size, self.concept_num, self.hidden_dim), device=features.device))
        pred_list = []
        ec_list = []  # concept embedding list in VAE
        rec_list = []  # reconstructed embedding list in VAE
        z_prob_list = []  # probability distribution of latent variable z in VAE
        for i in range(seq_len):
            xt = features[:, i]  # [batch_size]
            qt = questions[:, i]  # [batch_size]
            qt_mask = torch.ne(qt, -1)  # [batch_size], next_qt != -1
            tmp_ht = self._aggregate(xt, qt, ht, batch_size)  # [batch_size, concept_num, hidden_dim + embedding_dim]
            h_next, concept_embedding, rec_embedding, z_prob = self._update(tmp_ht, ht, qt)  # [batch_size, concept_num, hidden_dim]
            ht[qt_mask] = h_next[qt_mask]  # update new ht
            yt = self._predict(h_next, qt)  # [batch_size, concept_num]
            if i < seq_len - 1:
                pred = self._get_next_pred(yt, questions[:, i + 1])
                pred_list.append(pred)
            ec_list.append(concept_embedding)
            rec_list.append(rec_embedding)
            z_prob_list.append(z_prob)
        pred_res = torch.stack(pred_list, dim=1)  # [batch_size, seq_len - 1]
        return pred_res, ec_list, rec_list, z_prob_list


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module
    NOTE: Stole and modify from https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/SubLayers.py
    """

    def __init__(self, n_head, concept_num, input_dim, d_k, dropout=0.):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.concept_num = concept_num
        self.d_k = d_k
        self.w_qs = nn.Linear(input_dim, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(input_dim, n_head * d_k, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)
        # inferred latent graph, used for saving and visualization
        self.graphs = torch.zeros(n_head, concept_num, concept_num)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)

    def _get_graph(self, attn_score, qt):
        r"""
        Parameters:
            attn_score: attention score of all queries
            qt: masked question index
        Shape:
            attn_score: [n_head, mask_num, concept_num]
            qt: [mask_num]
        Return:
            graphs: n_head types of inferred graphs
        """
        graphs = Variable(torch.zeros(self.n_head, self.concept_num, self.concept_num))
        for k in range(self.n_head):
            index_tuple = (qt.long(), )
            graphs[k] = graphs[k].index_put(index_tuple, attn_score[k])  # used for calculation
            self.graphs[k] = self.graphs[k].index_put(index_tuple, attn_score[k])  # used for saving and visualization
        return graphs

    def forward(self, qt, query, key, mask=None):
        r"""
        Parameters:
            qt: masked question index
            query: answered concept embedding for a student batch
            key: concept embedding matrix
            mask: mask matrix
        Shape:
            qt: [mask_num]
            query: [mask_num, embedding_dim]
            key: [concept_num, embedding_dim]
        Return:
            graphs: n_head types of inferred graphs
        """
        d_k, n_head = self.d_k, self.n_head
        len_q, len_k = query.size(0), key.size(0)

        # Pass through the pre-attention projection: lq x (n_head *dk)
        # Separate different heads: lq x n_head x dk
        q = self.w_qs(query).view(len_q, n_head, d_k)
        k = self.w_ks(key).view(len_k, n_head, d_k)

        # Transpose for attention dot product: n_head x lq x dk
        q, k = q.transpose(0, 1), k.transpose(0, 1)
        attn_score = self.attention(q, k, mask=mask)  # [n_head, mask_num, concept_num]
        graphs = self._get_graph(attn_score, qt)
        return graphs


class VAE(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, msg_hidden_dim, msg_output_dim, concept_num, edge_type_num=2,
                 tau=0.1, factor=True, dropout=0., bias=True):
        super(VAE, self).__init__()
        self.edge_type_num = edge_type_num
        self.concept_num = concept_num
        self.tau = tau
        self.encoder = MLPEncoder(input_dim, hidden_dim, output_dim, factor=factor, dropout=dropout, bias=bias)
        self.decoder = MLPDecoder(input_dim, msg_hidden_dim, msg_output_dim, hidden_dim, edge_type_num, dropout=dropout, bias=bias)
        # inferred latent graph, used for saving and visualization
        self.graphs = torch.zeros(edge_type_num, concept_num, concept_num)

    def _get_graph(self, edges, rel_rec, rel_send):
        r"""
        Parameters:
            edges: sampled latent graph edge weights from the probability distribution of the latent variable z
            rel_rec: one-hot encoded receive-node index
            rel_send: one-hot encoded send-node index
        Shape:
            edges: [edge_num, edge_type_num]
            rel_rec: [edge_num, concept_num]
            rel_send: [edge_num, concept_num]
        Return:
            graphs: latent graph list modeled by z which has different edge types
        """
        x_index = torch.where(rel_send)[1].long()  # send node index: [edge_num, ]
        y_index = torch.where(rel_rec)[1].long()   # receive node index [edge_num, ]
        graphs = Variable(torch.zeros(self.edge_type_num, self.concept_num, self.concept_num))
        for k in range(self.edge_type_num):
            index_tuple = (x_index, y_index)
            graphs[k] = graphs[k].index_put(index_tuple, edges[:, k])  # used for calculation
            self.graphs[k] = self.graphs[k].index_put(index_tuple, edges[:, k])  # used for saving and visualization
        return graphs

    def forward(self, data, rel_send, rel_rec):
        r"""
        Parameters:
            data: input concept embedding matrix
            rel_send: one-hot encoded send-node index
            rel_rec: one-hot encoded receive-node index
        Shape:
            data: [concept_num, embedding_dim]
            rel_send: [edge_num, concept_num]
            rel_rec: [edge_num, concept_num]
        Return:
            graphs: latent graph list modeled by z which has different edge types
            output: the reconstructed data
            prob: q(z|x) distribution
        """
        logits = self.encoder(data, rel_rec, rel_send)  # [edge_num, output_dim(edge_type_num)]
        edges = gumbel_softmax(logits, tau=self.tau, dim=-1)  # [edge_num, edge_type_num]
        prob = F.softmax(logits, dim=-1)
        output = self.decoder(data, edges, rel_rec, rel_send)  # [concept_num, embedding_dim]
        graphs = self._get_graph(edges, rel_rec, rel_send)
        return graphs, output, prob


class DKT(nn.Module):

    def __init__(self, feature_dim, hidden_dim, output_dim, dropout=0., bias=True):
        super(DKT, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.bias = bias
        self.rnn = nn.LSTM(feature_dim, hidden_dim, bias=bias, dropout=dropout, batch_first=True)
        self.f_out = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, (nn.LSTM)):
                nn.init.orthogonal(m.weight)

    def _get_next_pred(self, yt, questions):
        r"""
        Parameters:
            y: predicted correct probability of all concepts at the next timestamp
            questions: question index matrix
        Shape:
            y: [batch_size, seq_len - 1, output_dim]
            questions: [batch_size, seq_len]
            pred: [batch_size, ]
        Return:
            pred: predicted correct probability of the question answered at the next timestamp
        """
        one_hot = torch.eye(self.output_dim, device=yt.device)
        one_hot = torch.cat((one_hot, torch.zeros(1, self.output_dim)), dim=0)
        next_qt = questions[:, 1:]
        next_qt = torch.where(next_qt != -1, next_qt, self.output_dim * torch.ones_like(next_qt, device=yt.device))  # [batch_size, seq_len - 1]
        one_hot_qt = F.embedding(next_qt, one_hot)  # [batch_size, seq_len - 1, output_dim]
        # dot product between yt and one_hot_qt
        pred = (yt * one_hot_qt).sum(dim=-1)  # [batch_size, seq_len - 1]
        return pred

    def forward(self, features, questions):
        r"""
        Parameters:
            features: input one-hot matrix
            questions: question index matrix
        seq_len dimension needs padding, because different students may have learning sequences with different lengths.
        Shape:
            features: [batch_size, seq_len]
            questions: [batch_size, seq_len]
            pred_res: [batch_size, seq_len - 1]
        Return:
            pred_res: the correct probability of questions answered at the next timestamp
            concept_embedding: input of VAE (optional)
            rec_embedding: reconstructed input of VAE (optional)
            z_prob: probability distribution of latent variable z in VAE (optional)
        """
        feat_one_hot = torch.eye(self.feature_dim)
        feat_one_hot = torch.cat((feat_one_hot, torch.zeros(1, self.feature_dim)), dim=0)
        feat = torch.where(features != -1, features, self.feature_dim * torch.ones_like(features, device=features.device))
        features = F.embedding(feat, feat_one_hot)

        feature_lens = torch.ne(questions, -1).sum(dim=1)  # padding value = -1
        x_packed = pack_padded_sequence(features, feature_lens, batch_first=True, enforce_sorted=False)
        output_packed, _ = self.rnn(x_packed)  # [batch, seq_len, hidden_dim]
        output_padded, output_lengths = pad_packed_sequence(output_packed, batch_first=True)  # [batch, seq_len, hidden_dim]
        yt = self.f_out(output_padded)  # [batch, seq_len, output_dim]
        yt = torch.sigmoid(yt)
        yt = yt[:, :-1, :]  # [batch, seq_len - 1, output_dim]
        pred_res = self._get_next_pred(yt, questions)  # [batch, seq_len - 1]
        return pred_res