import torch
from torch import nn
import torch.nn.functional as F


class SGNS(nn.Module):
    """ (Negative Sampling) Cross Entropy """
    def __init__(self):
        """"""
        super(SGNS, self).__init__()

    def forward(self, hv, targ, weight=None):
        """"""
        if weight is not None:
            return -(F.logsigmoid(hv * targ) * weight).mean()
        else:
            return -F.logsigmoid(hv * targ).mean()


class SGNSMSE(nn.Module):
    """ Dual loss combining MSE (for regression) and SGNS """
    def __init__(self, lmbda=0.5):
        """"""
        super(SGNSMSE, self).__init__()
        self.sgns = SGNS()
        self.mse = nn.MSELoss()
        self.lmbda = coeff

    def forward(self, h, v, q, targ, weight=None):
        """"""
        hv = torch.bmm(v, h.view(h.shape[0], h.shape[-1], 1)).squeeze()
        return (
            lmbda * self.sgns(hv, targ, weight) +
            (1.-lmbda) * self.mse(h, q)
        )

