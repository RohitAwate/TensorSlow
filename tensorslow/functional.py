import numpy as np
from .tensor import Tensor


def dot(a: Tensor, b: Tensor) -> Tensor:
    return a.dot(b)


def log(a: Tensor) -> Tensor:
    return a.log()


def exp(a: Tensor) -> Tensor:
    return a.exp()


HANDLED_FUNCS = {}


def implements(np_function):
    def decorator(func):
        HANDLED_FUNCS[np_function] = func
        return func

    return decorator

@implements(np.max)
def max(a: Tensor, **kwargs) -> Tensor:
    return Tensor(np.max(a.arr, **kwargs))

@implements(np.sum)
def sum(a: Tensor, **kwargs) -> Tensor:
    retval = Tensor(np.sum(a.arr, **kwargs))
    return retval
