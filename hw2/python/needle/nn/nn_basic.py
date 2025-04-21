"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from math import prod

class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.bias = bias
        self.weight = Parameter(
            init.kaiming_uniform(self.in_features, self.out_features, device=device, dtype=dtype)
        )
        # TODO why `fan_in`==`out_features`?
        if self.bias: 
            self.bias = Parameter(
                init.kaiming_uniform(fan_in=out_features, fan_out=1, device=device, dtype=dtype).transpose()
            )
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        X = X @ self.weight # B, n
        if self.bias:
            X = X + self.bias.broadcast_to((*X.shape[:-1], self.out_features))
        return X
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        B, C = X.shape[0], prod(X.shape[1:])
        return ops.reshape(X, (B, C))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        y_one_hot = init.one_hot(logits.shape[-1], y)  # Shape: (*batch_shape, num_classes)
        log_sum_exp = ops.logsumexp(logits, axes=(-1,))  # Shape: (*batch_shape,)
        true_logits = ops.summation(y_one_hot * logits, axes=(-1,))  # Shape: (*batch_shape,)
        return ops.summation(log_sum_exp - true_logits)/log_sum_exp.shape[0]  # Shape: (*batch_shape,)
        ### END YOUR SOLUTION

class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.running_mean= init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        B, C = x.shape
        if self.training:
            e = ops.summation(x, axes=(0, )) / B
            e_mat = ops.broadcast_to(e, x.shape)
            var = ops.summation((x - e_mat) ** 2, axes=(0, )) / (B)
            self.running_mean= (1 - self.momentum) * self.running_mean + self.momentum * e.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data
            x_hat = (x - e_mat) / ops.broadcast_to((var + self.eps) ** 0.5, x.shape)
            scale_x = x_hat * ops.broadcast_to(self.weight, x.shape)
            return scale_x + ops.broadcast_to(self.bias, x.shape)
        else:
            x_hat = (x - ops.broadcast_to(self.running_mean, x.shape)) / ops.broadcast_to(ops.power_scalar(self.running_var + self.eps, 0.5), x.shape)
            return x_hat * ops.broadcast_to(self.weight, x.shape) + ops.broadcast_to(self.bias, x.shape)
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        B, C = x.shape
        e = ops.summation(x, axes=(-1,)) / C
        e_mat = ops.broadcast_to(ops.reshape(e, (B, 1)), x.shape)
        var = ops.summation(ops.power_scalar(x - e_mat, 2), axes=(-1,)) / C
        x_hat = (x - e_mat) / ops.broadcast_to(ops.reshape(ops.power_scalar(var + self.eps, 0.5), (B, 1)), x.shape) # (B, C)
        # self.weight is scalar weights so just element-wise multiplication
        x = x_hat * ops.broadcast_to(self.weight, x.shape) + ops.broadcast_to(self.bias, x.shape)
        return x
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            x = (x * init.randb(*x.shape, p=1 - self.p, dtype="int8")) / (1-self.p)
            return x
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION
