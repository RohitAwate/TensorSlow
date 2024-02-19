from .module import Module
from ..tensor import Tensor

import numpy as np


class Dense(Module):
    def __init__(self, in_dim, out_dim, weight_scale=1e-3):
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.wdim = (in_dim, out_dim)
        self.W = Tensor(np.random.normal(scale=weight_scale, size=self.wdim))
        self.bdim = out_dim
        self.b = Tensor(np.zeros(out_dim))

    def forward(self, x):
        return x.dot(self.W) + self.b
