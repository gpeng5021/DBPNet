#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import numpy as np

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class L1_loss_zero(nn.Module):
    """L1 loss."""

    def __init__(self):
        super(L1_loss_zero, self).__init__()

    def forward(self, X, Y):
        Y_coor = torch.nonzero(Y, as_tuple=False)
        X = torch.take(X, Y_coor)
        Y = torch.take(Y, Y_coor)

        diff = torch.add(X, -Y)
        error = torch.abs(diff)
        loss = torch.sum(error)
        return loss
