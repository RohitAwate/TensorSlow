from typing import Optional, Union

import numpy as np

class Tensor:
    TensorInitializerType = Union[list, tuple, np.ndarray]

    def __init__(
        self, arr: TensorInitializerType, requires_grad: bool = True, _graph: dict = {}
    ):
        self.arr = np.array(arr) if type(arr) != np.ndarray else arr

        # Grad stuff
        self.requires_grad = requires_grad
        self.grad = np.zeros(self.arr.shape)
        self.grad_fn = lambda: pass
        self._graph = _graph

    def backward(self):
        assert self.requires_grad, "Tensor does not require grad"
        self.grad_fn()

    def __add__(self, other):
        graph = {"prev": (self, other), "op": "+"}
        out = Tensor(self.arr + other.arr, _graph=graph)

        def add_backward():
            self_local = 1.0
            other_local = 1.0
            upstream_grad = out.grad

            self.grad = self_local * upstream_grad
            other.grad = other_local * upstream_grad

        self.grad_fn = add_backward
        return out

    def __sub__(self, other):
        graph = {"prev": (self, other), "op": "-"}
        out = Tensor(self.arr - other.arr, _graph=graph)

        def sub_backward():
            self_local = 1.0
            other_local = -1.0
            upstream_grad = out.grad

            self.grad = self_local * upstream_grad
            other.grad = other_local * upstream_grad

        self.grad_fn = sub_backward
        return out

    def __mul__(self, other):
        graph = {"prev": (self, other), "op": "*"}
        out = Tensor(self.arr * other.arr, _graph=graph)

        def mul_backward():
            self_local = other.arr
            other_local = self.arr
            upstream_grad = out.grad

            self.grad = self_local * upstream_grad
            other.grad = other_local * upstream_grad

        self.grad_fn = mul_backward
        return out

    def __truediv__(self, other):
        graph = {"prev": (self, other), "op": "/"}
        out = Tensor(self.arr / other.arr, _graph=graph)

        def div_backward():
            self_local = 1 / other.arr
            other_local = np.negative(self.arr) / np.square(other.arr)
            upstream_grad = out.grad

            self.grad = self_local * upstream_grad
            other.grad = other_local * upstream_grad

        self.grad_fn = div_backward
        return out

    def __pow__(self, other):
        graph = {"prev": (self, other), "op": "**"}
        out = Tensor(np.power(self.arr, other.arr), _graph=graph)

        def pow_backward():
            self_local = np.power(self.arr, other.arr - 1) * out.grad
            other_local = out.arr * np.log(self.arr)
            upstream_grad = out.grad

            self.grad = self_local * upstream_grad
            other.grad = other_local * upstream_grad

        self.grad_fn = pow_backward
        return out

    def __getattr__(self, name):
        if hasattr(np.ndarray, name):
            attr = getattr(np.ndarray, name)
            bound_attr = attr.__get__(self.arr, type(self.arr))
            return bound_attr
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

    def __repr__(self):
        return f"Tensor{self.arr}"
