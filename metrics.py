import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from utils import nll_gaussian, kl_categorical, kl_categorical_uniform, accuracy

# Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network.
# For more information, please refer to https://dl.acm.org/doi/10.1145/3350546.3352513
# Author: jhljx
# Email: jhljx8918@gmail.com


class KTLoss(nn.Module):

    def __init__(self):
        super(KTLoss, self).__init__()

    def forward(self, pred_answers, real_answers):
        r"""
        Parameters:
            pred_answers: the correct probability of questions answered at the next timestamp
            real_answers: the real results(0 or 1) of questions answered at the next timestamp
        Shape:
            pred_answers: [batch_size, seq_len - 1]
            real_answers: [batch_size, seq_len]
        Return:
        """
        real_answers = real_answers[:, 1:]  # timestamp=1 ~ T
        # real_answers shape: [batch_size, seq_len - 1]
        # Here we can directly use nn.BCELoss, but this loss doesn't have ignore_index function
        answer_mask = torch.ne(real_answers, -1)
        pred_one, pred_zero = pred_answers, 1.0 - pred_answers  # [batch_size, seq_len - 1]

        # calculate auc and accuracy metrics
        try:
            y_true = real_answers[answer_mask].cpu().detach().numpy()
            y_pred = pred_one[answer_mask].cpu().detach().numpy()
            auc = roc_auc_score(y_true, y_pred)  # may raise ValueError
            output = torch.cat((pred_zero[answer_mask].reshape(-1, 1), pred_one[answer_mask].reshape(-1, 1)), dim=1)
            label = real_answers[answer_mask].reshape(-1, 1)
            acc = accuracy(output, label)
            acc = float(acc.cpu().detach().numpy())
        except ValueError as e:
            auc, acc = -1, -1

        # calculate NLL loss
        pred_one[answer_mask] = torch.log(pred_one[answer_mask])
        pred_zero[answer_mask] = torch.log(pred_zero[answer_mask])
        pred_answers = torch.cat((pred_zero.unsqueeze(dim=1), pred_one.unsqueeze(dim=1)), dim=1)
        # pred_answers shape: [batch_size, 2, seq_len - 1]
        nll_loss = nn.NLLLoss(ignore_index=-1)  # ignore masked values in real_answers
        loss = nll_loss(pred_answers, real_answers.long())
        return loss, auc, acc


class VAELoss(nn.Module):

    def __init__(self, concept_num, edge_type_num=2, prior=False, var=5e-5):
        super(VAELoss, self).__init__()
        self.concept_num = concept_num
        self.edge_type_num = edge_type_num
        self.prior = prior
        self.var = var

    def forward(self, ec_list, rec_list, z_prob_list, log_prior=None):
        time_stamp_num = len(ec_list)
        loss = 0
        for time_idx in range(time_stamp_num):
            output = rec_list[time_idx]
            target = ec_list[time_idx]
            prob = z_prob_list[time_idx]
            loss_nll = nll_gaussian(output, target, self.var)
            if self.prior:
                assert log_prior is not None
                loss_kl = kl_categorical(prob, log_prior, self.concept_num)
            else:
                loss_kl = kl_categorical_uniform(prob, self.concept_num, self.edge_type_num)
            if time_idx == 0:
                loss = loss_nll + loss_kl
            else:
                loss = loss + loss_nll + loss_kl
        return loss / time_stamp_num



