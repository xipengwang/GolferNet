import torch
import torch.nn as nn

class HeatmapLoss(nn.Module):
    """
    loss for detection heatmap
    """
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        loss = ((pred - gt)**2)
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1).mean(dim=0)
        return  loss

class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()

  def forward(self, pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
      Arguments:
       pred (batch x c x h x w)
       gt_regr (batch x c x h x w)
     '''
    # https://arxiv.org/pdf/1808.01244.pdf (1)
    alpha = 2
    beta = 4

    pos_inds = gt.eq(1).float()
    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    pos_loss = pos_loss.sum()

    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, beta)
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

    loss = 0

    num_pos  = pos_inds.float().sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss
