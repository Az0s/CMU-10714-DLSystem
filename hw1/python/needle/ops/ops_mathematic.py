"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

BACKEND = "np"
import numpy as array_api

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, b)
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad * rhs * power(lhs, rhs - 1), out_grad * log(lhs) * power(lhs, rhs)
        ### END YOUR SOLUTION

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # TODO: Y node.inputs is returned as tuple
        i = node.inputs[0]
        return out_grad * self.scalar * power_scalar(i, self.scalar-1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs 
        return out_grad * power_scalar(rhs, -1), negate(out_grad * lhs * power_scalar(rhs, -2))
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return divide_scalar(out_grad, self.scalar)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # kinda ugly. should improve
        if self.axes is None:
            self.axes = list(range(0, a.ndim, 1))
            self.axes[-1], self.axes[-2] = self.axes[-2], self.axes[-1]
        else:
            axes = list(range(0, a.ndim, 1))
            axes[self.axes[-1]], axes[self.axes[-2]] = axes[self.axes[-2]], axes[self.axes[-1]]
            self.axes = axes
        return array_api.transpose(a, axes=tuple(self.axes))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    """Transpose the input tensor.

    Args:
        a (_type_): _description_
        axes (_type_, optional): axes: tuple or list of ints, optional If specified, it must be a tuple or list which contains a permutation of [0, 1, ..., N-1] where N is the number of axes of a. Negative indices can also be used to specify axes. The i-th axis of the returned array will correspond to the axis numbered axes[i] of the input. If not specified, defaults to range(a.ndim)[::-1], which reverses the order of the axes, defaults to the last two axes.

    Returns:
        _type_: _description_
    """
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def reshape(a, shape):
    """_summary_

    Args:
        a (_type_): _description_
        shape (_type_): shape: int or tuple of ints The new shape should be compatible with the original shape. If an integer, then the result will be a 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.

    Returns:
        _type_: _description_
    """
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)  
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    """_summary_

    Args:
        a (_type_): _description_
        shape (_type_): tuple or int The shape of the desired array. A single integer i is interpreted as (i,).

    Returns:
        _type_: _description_
    """
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes, keepdims=False)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        """ NOTE: kinda tricky. should follow the steps below to clear things up. 
        Step 1: Define the Gradient Shape
            Input shape: Lets call it (d0, d1, ..., dn-1).
            Summed axes: S = set(self.axes) if self.axes is not None, otherwise S = {0, 1, ..., n-1}.
            Output shape: Input shape with axes in S removed, keeping the remaining axes sizes.
            Gradient shape: Same number of dimensions as the input, where:
            Size is 1 for axes in S (summed axes).
            Size is di (input size) for axes not in S (non-summed axes).
            Final gradient: Broadcast this to the input shape.

        Step 2: Reshape out_grad
            out_grad has the output shape (axes in S removed).
            We need to reshape it to the gradient shape by inserting singleton dimensions (1) at the positions of the summed axes.

        Step 3: Broadcast to Input Shape
            Broadcasting from the gradient shape to the input shape replicates out_grad along the summed axes, which is exactly what we want.
        """
        # sum over all when axes=None
        i = node.inputs[0]
        if self.axes is None:
            axes = tuple(range(len(i.shape)))
        else:
            axes = self.axes
        return broadcast_to(reshape(out_grad, tuple(1 if d in axes else i.shape[d] for d in range(len(i.shape)))), i.shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        """pretending they are scalar and piece shape together. 

        Args:
            out_grad (_type_): shape(m, k)
            node (_type_): 
                lhs: shape(m, n)
                rhs: shape(n, k)

        Returns:
            lhs_grad: (m, n)
            rhs_grad: (n, k)
        """
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lhs_grad = matmul(out_grad, transpose(rhs))
        rhs_grad = matmul(transpose(lhs), out_grad)
        return lhs_grad, rhs_grad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return negate(out_grad)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        i = node.inputs[0]
        return out_grad*power_scalar(i, -1)
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        i = node.inputs[0]
        return out_grad * exp(i)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # !FIX
        i: Value = node.inputs[0]
        return array_api.minimum(array_api.maximum(i.cached_data, 0), 1) * out_grad
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

