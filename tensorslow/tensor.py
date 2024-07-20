from typing import Union
from collections import deque

import numpy as np


class Tensor:
    TensorInitializerType = Union[list, tuple, np.ndarray]

    def __init__(
        self, arr: TensorInitializerType, requires_grad: bool = False, _graph: dict = {}
    ):
        self.arr = np.array(arr) if type(arr) != np.ndarray else arr

        # Grad stuff
        self.requires_grad = requires_grad
        self.grad = np.zeros(self.arr.shape)
        self.grad_fn = lambda: None
        self._graph = _graph

    def backward(self):
        assert self.requires_grad, "Tensor does not require grad"
        assert self._graph, "Tensor does not have a gradient graph"

        # Backpropagate gradients in reverse order
        self.grad = np.ones(self.grad.shape)
        for node in self._get_topsorted_graph():
            print(node.grad_fn)
            node.grad_fn()

    def __add__(self, other):
        graph = {"prev": (self, other), "op": "+"}
        out = Tensor(self.arr + other.arr, _graph=graph)

        def add_backward():
            self_local = np.ones(self.grad.shape)
            other_local = np.ones(other.grad.shape)
            upstream_grad = out.grad

            self.grad += self_local * upstream_grad
            other.grad += other_local * upstream_grad

        self.grad_fn = add_backward
        return out

    def __sub__(self, other):
        graph = {"prev": (self, other), "op": "-"}
        out = Tensor(self.arr - other.arr, _graph=graph)

        def sub_backward():
            self_local = 1.0
            other_local = -1.0
            upstream_grad = out.grad

            self.grad += self_local * upstream_grad
            other.grad += other_local * upstream_grad

        self.grad_fn = sub_backward
        return out

    def __mul__(self, other):
        graph = {"prev": (self, other), "op": "*"}
        out = Tensor(self.arr * other.arr, _graph=graph)

        def mul_backward():
            self_local = other.arr
            other_local = self.arr
            upstream_grad = out.grad

            self.grad += self_local * upstream_grad
            other.grad += other_local * upstream_grad

        self.grad_fn = mul_backward
        return out

    def __truediv__(self, other):
        graph = {"prev": (self, other), "op": "/"}
        out = Tensor(self.arr / other.arr, _graph=graph)

        """
        Problem is that self is (100, 10) but other is (100, 1).
        Broadcasting should take care of it.
        But, while calculating other_local, we use self and thus, it gets
        broadcasted to (100, 10). However, other's grad is (100, 1).
        Thus, we cannot update the gradient.
        """

        def truediv_backward():
            self_local = 1 / other.arr
            other_local = np.negative(self.arr) / np.square(other.arr)
            upstream_grad = out.grad

            self.grad += self_local * upstream_grad
            other.grad += other_local * upstream_grad

        self.grad_fn = truediv_backward
        return out

    def __neg__(self):
        graph = {"prev": (self,), "op": "neg"}
        out = Tensor(np.negative(self.arr), _graph=graph)

        def neg_backward():
            self_local = -1
            upstream_grad = out.grad

            self.grad += self_local * upstream_grad

        self.grad_fn = neg_backward
        return out

    def __pow__(self, other):
        graph = {"prev": (self, other), "op": "**"}
        out = Tensor(np.power(self.arr, other.arr), _graph=graph)

        def pow_backward():
            self_local = np.power(self.arr, other.arr - 1) * other.arr
            other_local = out.arr * np.log(self.arr)
            upstream_grad = out.grad

            self.grad += self_local * upstream_grad
            other.grad += other_local * upstream_grad

        self.grad_fn = pow_backward
        return out

    def dot(self, other):
        graph = {"prev": (self, other), "op": "dot"}
        out = Tensor(np.dot(self.arr, other.arr), _graph=graph)

        def dot_backward():
            self_local = other.arr
            other_local = self.arr
            upstream_grad = out.grad

            self.grad += np.dot(upstream_grad.T, other_local)
            other.grad += np.dot(self_local, upstream_grad.T)

        self.grad_fn = dot_backward
        return out

    def log(self):
        graph = {"prev": (self,), "op": "log"}
        out = Tensor(np.log(self.arr), _graph=graph)

        def log_backward():
            self_local = 1 / self.arr
            upstream_grad = out.grad

            self.grad += self_local * upstream_grad

        self.grad_fn = log_backward
        return out

    def exp(self):
        graph = {"prev": (self,), "op": "exp"}
        out = Tensor(np.exp(self.arr), _graph=graph)

        def exp_backward():
            self_local = out.arr
            upstream_grad = out.grad

            self.grad += self_local * upstream_grad

        self.grad_fn = exp_backward
        return out

    def transpose(self):
        return Tensor(self.arr.T, requires_grad=self.requires_grad, _graph=self._graph)

    @property
    def T(self):
        return self.transpose()

    def mean(self, **kwargs):
        return Tensor(
            self.arr.mean(**kwargs),
            requires_grad=self.requires_grad,
            _graph=self._graph,
        )

    def _get_topsorted_graph(self):
        topsorted_graph = []

        queue = deque([self])
        visited = set()

        while queue:
            curr = queue.popleft()

            if curr in visited:
                continue

            topsorted_graph.append(curr)
            visited.add(curr)

            prevs = curr._graph.get("prev", [])
            for prev in prevs:
                queue.append(prev)

        return topsorted_graph

    def __array__(self):
        return self.arr

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__":
            unwrapped_inputs = []
            for input in inputs:
                if isinstance(input, Tensor):
                    unwrapped_inputs.append(input.arr)
                else:
                    unwrapped_inputs.append(input)
            return Tensor(ufunc(*unwrapped_inputs, **kwargs))
        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        from .functional import HANDLED_FUNCS

        if func not in HANDLED_FUNCS:
            return NotImplemented

        return HANDLED_FUNCS[func](*args, **kwargs)

    def __getitem__(self, key):
        return Tensor(self.arr.__getitem__(key))

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
        return f"Tensor<{self.arr}>"
