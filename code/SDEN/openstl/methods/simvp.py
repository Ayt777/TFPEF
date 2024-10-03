import torch
from openstl.models import SimVP_Model
from .base_method import Base_method
from openstl.modules.wast_modules import *
import torch.nn.functional as F
import math

class CustomMSELoss(nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()

    def forward(self, pred_y, batch_y):
        B, T, C, H, W = pred_y.shape

        weights = torch.arange(1, T+1).float().to(pred_y.device)

        weights = weights.view(1, T, 1, 1, 1).expand_as(pred_y)

        loss = (weights * (pred_y - batch_y) ** 2).mean()

        return loss

class SimVP(Base_method):
    r"""SimVP

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

    def __init__(self, **args):
        super().__init__(**args)

    def _build_model(self, **args):
        return SimVP_Model(**args)

    def forward(self, batch_x, batch_y=None, **kwargs):
        pre_seq_length, aft_seq_length = self.hparams.pre_seq_length, self.hparams.aft_seq_length
        if aft_seq_length == pre_seq_length:
            pred_y = self.model(batch_x)
        elif aft_seq_length < pre_seq_length:
            pred_y = self.model(batch_x)
            pred_y = pred_y[:, :aft_seq_length]
        elif aft_seq_length > pre_seq_length:
            pred_y = []
            d = aft_seq_length // pre_seq_length
            m = aft_seq_length % pre_seq_length
            
            cur_seq = batch_x.clone()
            for _ in range(d):
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq)
            if m != 0:
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq[:, :m])
            
            pred_y = torch.cat(pred_y, dim=1)
        # pred_y = self.model(batch_x)
        return pred_y

    # def forward(self, batch_x, batch_y=None, **kwargs):
    #     pre_seq_length, aft_seq_length = self.hparams.pre_seq_length, self.hparams.aft_seq_length

    #     pred_y = []

    #     cur_seq = batch_x.clone()

    #     for _ in range(aft_seq_length):
    #         cur_output = self.model(cur_seq)
    #         cur_output = cur_output.unsqueeze(1)
    #         pred_y.append(cur_output)
    #         cur_seq = torch.cat((cur_seq[:, 1:], cur_output), dim=1)

    #     pred_y = torch.cat(pred_y, dim=1)
    #     return pred_y

    

    def diff_div_reg(self, pred_y, batch_y, tau=0.01, eps=1e-12):
        B, T, C = pred_y.shape[:3]
        if T <= 2:  return 0
        gap_pred_y = (pred_y[:, 1:] - pred_y[:, :-1]).reshape(B, T-1, -1)
        gap_batch_y = (batch_y[:, 1:] - batch_y[:, :-1]).reshape(B, T-1, -1)
        softmax_gap_p = F.softmax(gap_pred_y / tau, -1)
        softmax_gap_b = F.softmax(gap_batch_y / tau, -1)
        loss_gap = torch.abs(softmax_gap_p - softmax_gap_b)
        return loss_gap.mean()
    
    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x)
        current_epoch = self.current_epoch
        # 4->4
        alpha = math.exp(-1*current_epoch/10)
        loss = self.criterion(pred_y, batch_y)
        loss = loss + alpha * 10 * self.diff_div_reg(pred_y, batch_y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
