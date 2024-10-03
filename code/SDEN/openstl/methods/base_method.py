import numpy as np
import torch.nn as nn
import os.path as osp
import pytorch_lightning as pl
from openstl.utils import print_log, check_dir
from openstl.core import get_optim_scheduler
from openstl.core import metric
import math
import numpy as np
import torch


class CustomMSELoss(nn.Module):
    def __init__(self, scale_factor=0.5):
        super(CustomMSELoss, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, pred_y, batch_y):
        B, T, C, H, W = pred_y.shape

        weights = torch.arange(1, T+1).float().to(pred_y.device)
        # weights = torch.exp(torch.arange(T).float() * self.scale_factor).to(pred_y.device)

        weights = weights.view(1, T, 1, 1, 1).expand_as(pred_y)

        loss = (weights * (pred_y - batch_y) ** 2).mean()

        return loss

class Base_method(pl.LightningModule):

    def __init__(self, **args):
        super().__init__()

        if 'weather' in args['dataname']:
            self.metric_list, self.spatial_norm = args['metrics'], True
            self.channel_names = args.data_name if 'mv' in args['data_name'] else None
        else:
            self.metric_list, self.spatial_norm, self.channel_names = args['metrics'], False, None

        self.save_hyperparameters()
        self.model = self._build_model(**args)
        self.criterion = CustomMSELoss()
        # self.criterion = nn.MSELoss()
        self.test_outputs = []

    def _build_model(self):
        raise NotImplementedError
    
    def configure_optimizers(self):
        optimizer, scheduler, by_epoch = get_optim_scheduler(
            self.hparams, 
            self.hparams.epoch, 
            self.model, 
            self.hparams.steps_per_epoch
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "epoch" if by_epoch else "step"
            },
        }

    def forward(self, batch):
        NotImplementedError
    
    def training_step(self, batch, batch_idx):
        NotImplementedError

    
    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)
        current_epoch = self.current_epoch
        # 4->4
        alpha = math.exp(-1*current_epoch/10)
        loss = self.criterion(pred_y, batch_y)
        loss = loss + alpha * 10 * self.diff_div_reg(pred_y, batch_y)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss
    
    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)
        mean,std = self.hparams.test_mean, self.hparams.test_std
        batch_x = batch_x.cpu() * std + mean
        pred_y = pred_y.cpu() * std + mean
        batch_y = batch_y.cpu() * std + mean
        outputs = {'inputs': batch_x.cpu().numpy(), 'preds': pred_y.cpu().numpy(), 'trues': batch_y.cpu().numpy()}
        self.test_outputs.append(outputs)
        return outputs

    def on_test_epoch_end(self):
        results_all = {}
        for k in self.test_outputs[0].keys():
            results_all[k] = np.concatenate([batch[k] for batch in self.test_outputs], axis=0)
        
        eval_res, eval_log = metric(results_all['preds'], results_all['trues'],
            self.hparams.test_mean, self.hparams.test_std, metrics=self.metric_list, 
            channel_names=self.channel_names, spatial_norm=self.spatial_norm)
        
        results_all['metrics'] = np.array([eval_res['mae'], eval_res['mse']])

        if self.trainer.is_global_zero:
            print_log(eval_log)
            folder_path = check_dir(osp.join(self.hparams.save_dir, 'saved'))

            for np_data in ['metrics', 'inputs', 'trues', 'preds']:
                np.save(osp.join(folder_path, np_data + '.npy'), results_all[np_data])
        return results_all