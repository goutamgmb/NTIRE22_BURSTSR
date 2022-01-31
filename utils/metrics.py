import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.warp import warp


class L2(nn.Module):
    def __init__(self, boundary_ignore=None):
        super().__init__()
        self.boundary_ignore = boundary_ignore

    def forward(self, pred, gt, valid=None):
        if self.boundary_ignore is not None:
            pred = pred[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            if valid is not None:
                valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        pred_m = pred
        gt_m = gt

        if valid is None:
            mse = F.mse_loss(pred_m, gt_m)
        else:
            mse = F.mse_loss(pred_m, gt_m, reduction='none')

            eps = 1e-12
            elem_ratio = mse.numel() / valid.numel()
            mse = (mse * valid.float()).sum() / (valid.float().sum()*elem_ratio + eps)

        return mse


class PSNR(nn.Module):
    def __init__(self, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.l2 = L2(boundary_ignore=boundary_ignore)
        self.max_value = max_value

    def psnr(self, pred, gt, valid=None):
        mse = self.l2(pred, gt, valid=valid)

        psnr = 20 * math.log10(self.max_value) - 10.0 * mse.log10()

        return psnr

    def forward(self, pred, gt, valid=None):
        assert pred.dim() == 4 and pred.shape == gt.shape
        if valid is None:
            psnr_all = [self.psnr(p.unsqueeze(0), g.unsqueeze(0)) for p, g in
                        zip(pred, gt)]
        else:
            psnr_all = [self.psnr(p.unsqueeze(0), g.unsqueeze(0), v.unsqueeze(0)) for p, g, v in zip(pred, gt, valid)]
        psnr = sum(psnr_all) / len(psnr_all)
        return psnr
