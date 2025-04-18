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
        assert a.shape == b.shape, (
        f"Shape mismatch: a.shape={a.shape}, b.shape={b.shape}. "
        "ndl does not support implicit shape broadcasting."
    )
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
        if len(a.shape) == 1:
            return a
        elif self.axes is None:
            self.axes = (-1, -2)
        axes = list(range(0, a.ndim, 1))
        axes[self.axes[-1]], axes[self.axes[-2]] = axes[self.axes[-2]], axes[self.axes[-1]]
        return array_api.transpose(a, axes=tuple(axes))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        i = node.inputs[0]
        if len(i.shape) == 1:
            return out_grad
        # self.axes should already be valid 
        return transpose(out_grad, axes=tuple(self.axes))
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
        return reshape(out_grad, node.inputs[0].shape)
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
        i = node.inputs[0]
        i_shape, out_shape = i.shape, out_grad.shape
        if len(i_shape) != len(out_shape):
            sum_axes = tuple(range(len(out_shape) - len(i_shape)))
            out_grad = summation(out_grad, axes=sum_axes)
            sum_axes = tuple([i for i in range(len(i_shape)) if i_shape[i] == 1 and out_grad.shape[i] != 1])
            if sum_axes:
                out_grad = reshape(summation(out_grad, sum_axes), i_shape)
        else: 
            sum_axes = tuple([i for i in range(len(i_shape)) if i_shape[i] == 1 and out_shape[i] != 1])
            if sum_axes:
                out_grad = reshape(summation(out_grad, sum_axes), i_shape)
        return out_grad  
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
        # `int` is not iterable will be raised if axes is an int
        if axes is not None:
            if isinstance(axes, int):
                axes = (axes,)
            if not all(isinstance(a, int) for a in axes):
                raise TypeError("axes should be a tuple of integers")
            axes = tuple(sorted(axes))
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
        
        # scalar
        if len(i.shape) == 0:
            return out_grad
        if self.axes is None:
            axes = tuple(range(len(i.shape)))
        else:
            if any(x<0 for x in self.axes):
                self.axes = tuple(x if x>=0 else x + len(i.shape) for x in self.axes)
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
            out_grad (_type_): shape(..., m, k)
            node (_type_): 
                lhs: shape(..., m, n)
                rhs: shape(..., n, k)
        NOTE
        In batched matrix multiplication, A or B may have fewer batch dimensions or dimensions of size 1, which are broadcasted to match the output's batch dimensions. The gradients grad_a and grad_b will have the broadcasted batch shape (same as out_grad's batch dimensions), but they must be reduced to A's and B's original shapes by summing over broadcasted dimensions.
        
        For grad_a: If A has fewer dimensions than out_grad (e.g., A is (m, k) while out_grad is (batch, m, n)), sum over the extra leading batch dimensions. If A has the same number of dimensions but some batch dimensions are 1 (e.g., A is (1, m, k) while out_grad is (batch, m, n)), sum over those dimensions where A.shape[i] == 1 and out_grad.shape[i] > 1.
        
        For grad_b: Apply the same logic.

        Returns:
            lhs_grad: (..., m, n)
            rhs_grad: (..., n, k)
        """
        ### BEGIN YOUR SOLUTION
        # broadcasted into another batch dim
        lhs, rhs = node.inputs
        lhs_shape, rhs_shape, out_shape =lhs.shape, rhs.shape, out_grad.shape
        lhs_grad, rhs_grad = matmul(out_grad, transpose(rhs, axes=(-2, -1))), matmul(transpose(lhs, axes=(-2, -1)), out_grad)
        if len(lhs_shape) < len(out_shape):
            sum_axes = tuple(range(len(out_shape)-len(lhs_shape)))
            lhs_grad = summation(lhs_grad, axes=sum_axes)
        elif lhs_shape == out_shape:
            # if there's broadcasted batch dim with val 1
            sum_axes = tuple([i for i in range(len(lhs_shape) - 2) if lhs_shape[i] == 1 and out_shape[i] != 1])
            lhs_grad = reshape(summation(lhs_grad, axes=sum_axes), lhs_shape) # apply reshape since keepdim default is False
        # exactly the same as lhs
        if len(rhs_shape) < len(out_shape):
            sum_axes = tuple(range(len(out_shape) - len(rhs_shape)))
            rhs_grad = summation(rhs_grad, axes=sum_axes)
        elif rhs_shape == out_shape:
            sum_axes = tuple([i for i in range(len(rhs_shape) - 2) if rhs_shape[i] == 1 and out_shape[i] != 1])
            rhs_grad = reshape(summation(rhs_grad, axes=sum_axes), rhs_shape)
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
        i = node.inputs[0]
        # relu_gates = array_api.where(i.realize_cached_data() > 0, 1, 0).astype(array_api.int8)
        # return Tensor(array_api.multiply(out_grad.realize_cached_data(), relu_gates))
        return Tensor(array_api.where(i.realize_cached_data() > 0, out_grad.realize_cached_data(), 0))
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

