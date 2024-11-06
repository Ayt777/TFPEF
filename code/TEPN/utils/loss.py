from torch.nn.modules.loss import _Loss
from skimage.metrics import structural_similarity
import torch
import math
import torch.nn as nn
import numpy as np
class SSIM(object):
    def __init__(self):
        super(SSIM, self).__init__()
        self.rgb = 255.0

    def forward(self, x, y):
        batch_size = 5
        x = x.squeeze().permute(0, 2, 3, 1)
        y = y.squeeze().permute(0, 2, 3, 1)
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        ssim = 0
        for i in range(batch_size):
            img_a = (x[i] * 0.5 + 0.5) * self.rgb
            img_b = (y[i] * 0.5 + 0.5) * self.rgb
            img_a = img_a.round()
            img_b = img_b.round()
            ssim_ = structural_similarity(img_a, img_b, multichannel=True)
            ssim += ssim_

        ssim = ssim / batch_size
        return ssim


class relative_loss(_Loss):
    def __init__(self):
        super(relative_loss, self).__init__()
        # self.k = 1e-2
        self.k = 1e-2
        self.eps = 1e-10

    def forward(self, x, y):
        diff = torch.add(x, -y)
        # x ->input
        error = torch.sqrt(diff * diff+self.eps) / (x + self.k)
        # error = torch.sqrt(diff * diff) / (x + self.k)
        loss = torch.mean(error)
        return loss


class L1_Charbonnier_loss(_Loss):
    """
    L1 Charbonnierloss
    """

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps * self.eps)
        loss = torch.mean(error)
        return loss

class reprojection_loss(_Loss):
    # 相对误差+mse误差
    def __init__(self):
        super(reprojection_loss, self).__init__()
        self.l1_loss = nn.L1Loss()
    def forward(self,X,Y):
        loss = self.l1_loss(X,Y)
        return loss



class PSNR(object):
    def __init__(self):
        super(PSNR, self).__init__()
        self.rgb = 255.0

    def forward(self, x, y):
        x = (x.detach().cpu().numpy() * 0.5 + 0.5) * self.rgb
        y = (y.detach().cpu().numpy() * 0.5 + 0.5) * self.rgb
        x = x.round()
        y = y.round()
        mse = np.mean((x - y) ** 2) + 0.0000001
        psnr = 20 * math.log10(self.rgb / math.sqrt(mse))
        return psnr
