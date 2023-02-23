from typing import Optional, Union

import numpy as np


class Tensor:
    TensorInitializerType = Union[list, tuple, np.ndarray]

    def __init__(
            self, arr: TensorInitializerType, requires_grad: bool = True, _graph: dict = {}
    ):
        self.arr = np.array(arr) if type(arr) != np.ndarray else arr
        self.requires_grad = requires_grad
        self.grad = np.zeros(self.arr.shape)
        self._graph = _graph

    def backward(self):
        self.grad_fn()

    def grad_fn(self, upstream_grad: Optional[np.ndarray] = None, local_grad=None):
        assert self.requires_grad, "Tensor does not require grad"

        if upstream_grad is None and local_grad is None:
            upstream_grad = local_grad = np.ones(self.arr.shape)

        assert upstream_grad.shape == local_grad.shape

        self.grad = np.multiply(upstream_grad, local_grad)

        if not self._graph:
            return

        op = self._graph["op"]
        prev1, prev2 = self._graph["prev"]

        if op == "+":
            local_grads = (np.ones(prev1.arr.shape), np.ones(prev2.arr.shape))
        elif op == "-":
            local_grads = (np.ones(prev1.arr.shape), -np.ones(prev2.arr.shape))
        elif op == "*":
            local_grads = (prev2.arr, prev1.arr)
        elif op == "/":
            num_local_grad = 1 / prev2.arr
            den_local_grad = -prev1.arr / np.square(prev2.arr)
            local_grads = (num_local_grad, den_local_grad)
        else:
            raise ValueError("Invalid operator: " + op)

        prev1_local_grad, prev2_local_grad = local_grads
        prev1.grad_fn(upstream_grad=self.grad, local_grad=prev1_local_grad)
        prev2.grad_fn(upstream_grad=self.grad, local_grad=prev2_local_grad)

    def __getattr__(self, name):
        if hasattr(np.ndarray, name):
            attr = getattr(np.ndarray, name)
            bound_attr = attr.__get__(self.arr, type(self.arr))
            return bound_attr
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

    def __add__(self, other):
        graph = {"prev": (self, other), "op": "+"}
        return Tensor(self.arr + other.arr, _graph=graph)

    def __sub__(self, other):
        graph = {"prev": (self, other), "op": "-"}
        return Tensor(self.arr - other.arr, _graph=graph)

    def __mul__(self, other):
        graph = {"prev": (self, other), "op": "*"}
        return Tensor(self.arr * other.arr, _graph=graph)

    def __truediv__(self, other):
        graph = {"prev": (self, other), "op": "/"}
        return Tensor(self.arr / other.arr, _graph=graph)

    def __pow__(self, other):
        graph = {"prev": (self, other), "op": "**"}
        return Tensor(np.power(self.arr, other.arr), _graph=graph)

    def __repr__(self):
        return f"Tensor{self.arr}"
