from .module import Module
from ..tensor import Tensor
from ..functional import *

import numpy as np


class Dense(Module):
    def __init__(self, in_dim, out_dim, weight_scale=1e-3):
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.wdim = (in_dim, out_dim)
        self.W = Tensor(
            np.random.normal(scale=weight_scale, size=self.wdim), requires_grad=True
        )
        self.bdim = out_dim
        self.b = Tensor(np.zeros(out_dim), requires_grad=True)

    def forward(self, x):
        return x.dot(self.W) + self.b

    def backward(self):
        return super().backward()


class Softmax(Module):
    def forward(self, x):
        e_x = exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)
