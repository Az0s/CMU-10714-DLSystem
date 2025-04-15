from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        self.axes = (1,)
        return Z - array_api.broadcast_to(
            array_api.reshape(
                logsumexp(Tensor(Z), self.axes).realize_cached_data(),
                (*Z.shape[:-1], 1),
            ),
            Z.shape,
        )
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        Z = node.inputs[0].realize_cached_data()
        maxZ = array_api.max(Z, self.axes)
        maxZmatShape = tuple(
            [1 if d in self.axes else Z.shape[d] for d in range(len(Z.shape))]
        )
        stableExpZ = array_api.exp(
            Z - array_api.broadcast_to(array_api.reshape(maxZ, maxZmatShape), Z.shape)
        )
        sumExp = array_api.sum(stableExpZ, self.axes)
        s = stableExpZ / array_api.broadcast_to(
            array_api.reshape(sumExp, maxZmatShape), Z.shape
        )
        # TODO why sum out_grad?
        sum_out_grad = array_api.sum(out_grad.realize_cached_data(), self.axes)
        sum_out_grad_broadcasted = array_api.broadcast_to(
            array_api.reshape(sum_out_grad, maxZmatShape), Z.shape
        )
        grad = out_grad.realize_cached_data() - s * sum_out_grad_broadcasted
        return Tensor(grad)


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
        self.stableExpZ = None
        self.sumExp = None
        self.maxZmatShape = None

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        if not self.axes:
            self.axes = tuple(range(len(Z.shape)))
        if any(x < 0 for x in self.axes):
            self.axes = tuple(x if x >= 0 else len(Z.shape) + x for x in self.axes)
        maxZ = array_api.max(Z, self.axes)
        self.maxZmatShape = tuple(
            [1 if d in self.axes else Z.shape[d] for d in range(len(Z.shape))]
        )
        self.stableExpZ = array_api.exp(
            Z
            - array_api.broadcast_to(
                array_api.reshape(maxZ, self.maxZmatShape), Z.shape
            )
        )
        self.sumExp = array_api.sum(self.stableExpZ, self.axes)
        return array_api.log(self.sumExp) + maxZ
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # NOTE the numerically stable log-sum-exp function formula shouldn't affect its derivative
        ### BEGIN YOUR SOLUTION
        return Tensor(
            self.stableExpZ
            * array_api.broadcast_to(
                array_api.reshape(
                    out_grad.realize_cached_data() / self.sumExp, self.maxZmatShape
                ),
                self.stableExpZ.shape,
            )
        )
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
