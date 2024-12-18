import numpy as np
from .tensor import Tensor


def dot(a: Tensor, b: Tensor) -> Tensor:
    return a.dot(b)


def log(a: Tensor) -> Tensor:
    return a.log()


def exp(a: Tensor) -> Tensor:
    return a.exp()


def maximum(a: Tensor, b: Tensor) -> Tensor:
    return Tensor(np.maximum(a.arr, b.arr))


def max(a: Tensor, **kwargs) -> Tensor:
    return Tensor(np.max(a.arr, **kwargs))


def sum(a: Tensor, **kwargs) -> Tensor:
    retval = Tensor(np.sum(a.arr, **kwargs))
    print(a.shape, retval.shape, retval.grad.shape)
    return retval
