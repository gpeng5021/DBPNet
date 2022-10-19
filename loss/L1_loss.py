#!/usr/bin/env python
# encoding: utf-8


import torch
import torch.nn as nn
import numpy as np

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class L1_loss(nn.Module):
    """L1 loss."""

    def __init__(self):
        super(L1_loss, self).__init__()

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.abs(diff)
        loss = torch.sum(error)
        return loss
