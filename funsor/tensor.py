
import functools
import itertools
import typing
import warnings
from collections import Counter, OrderedDict
from contextlib import contextmanager
from functools import reduce

import numpy as np
import opt_einsum
from multipledispatch import dispatch

import funsor

from . import ops
from .delta import Delta
from .domains import Array, ArrayType, Bint, Product, Real, Reals, find_domain
from .ops import BinaryOp, FinitaryOp, GetitemOp, MatmulOp, Op, ReshapeOp
from .terms import (
    Binary,
    Finitary,
    Funsor,
    FunsorMeta,
    Lambda,
    Number,
    Scatter,
    Slice,
    Tuple,
    Unary,
    Variable,
    eager,
    substitute,
    to_data,
    to_funsor,
)
from .typing import Variadic
from .util import (
    as_callable,
    get_backend,
    get_tracing_state,
    getargspec,
    is_nn_module,
    quote,
)


def get_default_prototype():
    backend = get_backend()
    if backend == "torch":
        import torch

        return torch.tensor([])
    else:
        return np.array([])


def numeric_array(x, dtype=None, device=None):
    backend = get_backend()
    if backend == "torch":
        import torch

        return torch.tensor(x, dtype=dtype, device=device)
    else:
        return np.array(x, dtype=dtype)


def dummy_numeric_array(domain):
    value = 0.1 if domain.dtype == "real" else 1
    return ops.expand(numeric_array(value), domain.shape) if domain.shape else value


def _nameof(fn):
    return getattr(fn, "__name__", type(fn).__name__)


@contextmanager
def ignore_jit_warnings():
    if get_backend() != "torch":
        yield
        return

    import torch

    if not torch._C._get_tracing_state():
        yield
        return

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", "Iterating over a tensor")
        yield


class TensorMeta(FunsorMeta):

    def __call__(cls, data, inputs=None, dtype="real"):
        if inputs is None:
            inputs = tuple()
        elif isinstance(inputs, dict):
            inputs = tuple(inputs.items())
        if isinstance(data, np.generic):
            data = data.__array__()

        return super(TensorMeta, cls).__call__(data, inputs, dtype)


class Tensor(Funsor, metaclass=TensorMeta):

    def __init__(self, data, inputs=None, dtype="real"):
        assert ops.is_numeric_array(data)
        assert isinstance(inputs, tuple)
        if not get_tracing_state():
            assert len(inputs) <= len(data.shape)
            for (k, d), size in zip(inputs, data.shape):
                assert d.dtype == size
        inputs = OrderedDict(inputs)
        output = Array[dtype, data.shape[len(inputs) :]]
        fresh = frozenset(inputs.keys())
        bound = {}
        super(Tensor, self).__init__(inputs, output, fresh, bound)
        self.data = data

    @ignore_jit_warnings()
    def __repr__(self):
        data = repr(self.data).replace("\n", "\n       ")
        inputs = dict.__repr__(self.inputs)
        if self.dtype != "real":
            return "Tensor({}, {}, {})".format(data, inputs, repr(self.dtype))
        elif self.inputs:
            return "Tensor({}, {})".format(data, inputs)
        else:
            return "Tensor({})".format(data)

    @ignore_jit_warnings()
    def __str__(self):
        data = str(self.data).replace("\n", "\n       ")
        inputs = dict.__repr__(self.inputs)
        if self.dtype != "real":
            return "Tensor({}, {}, {})".format(data, inputs, repr(self.dtype))
        elif self.inputs:
            return "Tensor({}, {})".format(data, inputs)
        else:
            return data

    def __int__(self):
        return int(self.data)

    def __float__(self):
        return float(self.data)

    def __bool__(self):
        return bool(self.data)

    def item(self):
        return self.data.item()

    def clamp_finite(self):
        pass

    @property
    def requires_grad(self):
        pass

    def align(self, names):
        assert isinstance(names, tuple)
        assert all(name in self.inputs for name in names)
        if not names or names == tuple(self.inputs):
            return self

        inputs = OrderedDict((name, self.inputs[name]) for name in names)
        inputs.update(self.inputs)
        old_dims = tuple(self.inputs)
        new_dims = tuple(inputs)
        permutation = tuple(old_dims.index(d) for d in new_dims)
        permutation = permutation + tuple(
            range(len(permutation), len(permutation) + len(self.output.shape))
        )
        data = ops.permute(self.data, permutation)
        return Tensor(data, inputs, self.dtype)

    def eager_subs(self, subs):
        pass

    def eager_unary(self, op):
        pass

    def eager_reduce(self, op, reduced_vars):
        pass

    def _sample(self, sampled_vars, sample_inputs, rng_key):
        pass

    def new_arange(self, name, *args, **kwargs):
        """
        Helper to create a named :func:`torch.arange` or :func:`np.arange` funsor.
        In some cases this can be replaced by a symbolic
        :class:`~funsor.terms.Slice` .

        :param str name: A variable name.
        :param int start:
        :param int stop:
        :param int step: Three args following :py:class:`slice` semantics.
        :param int dtype: An optional bounded integer type of this slice.
        :rtype: Tensor
        """
        start = 0
        step = 1
        dtype = None
        if len(args) == 1:
            stop = args[0]
            dtype = kwargs.pop("dtype", stop)
        elif len(args) == 2:
            start, stop = args
            dtype = kwargs.pop("dtype", stop)
        elif len(args) == 3:
            start, stop, step = args
            dtype = kwargs.pop("dtype", stop)
        elif len(args) == 4:
            start, stop, step, dtype = args
        else:
            raise ValueError
        if step <= 0:
            raise ValueError
        stop = min(dtype, max(start, stop))
        data = ops.new_arange(self.data, start, stop, step)
        inputs = OrderedDict([(name, Bint[len(data)])])
        return Tensor(data, inputs, dtype=dtype)

    def materialize(self, x):
        """
        Attempt to convert a Funsor to a :class:`~funsor.terms.Number` or
        :class:`Tensor` by substituting :func:`arange` s into its free variables.

        :arg Funsor x: A funsor.
        :rtype: Funsor
        """
        assert isinstance(x, Funsor)
        if isinstance(x, (Number, Tensor)):
            return x
        subs = []
        for name, domain in x.inputs.items():
            if isinstance(domain.dtype, int):
                subs.append((name, self.new_arange(name, domain.dtype)))
        subs = tuple(subs)
        return substitute(x, subs)


@to_funsor.register(np.ndarray)
@to_funsor.register(np.generic)
def tensor_to_funsor(x, output=None, dim_to_name=None):
    if not dim_to_name:
        output = output if output is not None else Reals[x.shape]
        result = Tensor(x, dtype=output.dtype)
        if result.output != output:
            raise ValueError(
                "Invalid shape: expected {}, actual {}".format(
                    output.shape, result.output.shape
                )
            )
        return result
    else:
        assert all(
            isinstance(k, int) and k < 0 and isinstance(v, str)
            for k, v in dim_to_name.items()
        )

        if output is None:
            batch_ndims = min(-min(dim_to_name.keys()), len(x.shape))
            output = Reals[x.shape[batch_ndims:]]

        packed_inputs = OrderedDict()
        for dim, size in zip(range(len(x.shape) - len(output.shape)), x.shape):
            name = dim_to_name.get(dim + len(output.shape) - len(x.shape), None)
            if name is not None and size != 1:
                packed_inputs[name] = Bint[size]
        shape = tuple(d.size for d in packed_inputs.values()) + output.shape
        if x.shape != shape:
            x = x.reshape(shape)
        return Tensor(x, packed_inputs, dtype=output.dtype)


def align_tensor(new_inputs, x, expand=False):
    r"""
    Permute and add dims to a tensor to match desired ``new_inputs``.

    :param OrderedDict new_inputs: A target set of inputs.
    :param funsor.terms.Funsor x: A :class:`Tensor` or
        :class:`~funsor.terms.Number` .
    :param bool expand: If False (default), set result size to 1 for any input
        of ``x`` not in ``new_inputs``; if True expand to ``new_inputs`` size.
    :return: a number or :class:`torch.Tensor` or :class:`np.ndarray` that can be broadcast to other
        tensors with inputs ``new_inputs``.
    :rtype: int or float or torch.Tensor or np.ndarray
    """
    assert isinstance(new_inputs, OrderedDict)
    assert isinstance(x, (Number, Tensor))
    assert all(isinstance(d.dtype, int) for d in x.inputs.values())

    data = x.data
    if isinstance(x, Number):
        return data

    old_inputs = x.inputs
    if old_inputs == new_inputs:
        return data

    x_keys = tuple(old_inputs)
    data = ops.permute(
        data,
        tuple(x_keys.index(k) for k in new_inputs if k in old_inputs)
        + tuple(range(len(old_inputs), len(data.shape))),
    )

    data = data.reshape(
        tuple(old_inputs[k].dtype if k in old_inputs else 1 for k in new_inputs)
        + x.output.shape
    )

    if expand:
        data = ops.expand(
            data, tuple(d.dtype for d in new_inputs.values()) + x.output.shape
        )
    return data


def align_tensors(*args, **kwargs):
    r"""
    Permute multiple tensors before applying a broadcasted op.

    This is mainly useful for implementing eager funsor operations.

    :param funsor.terms.Funsor \*args: Multiple :class:`Tensor` s and
        :class:`~funsor.terms.Number` s.
    :param bool expand: Whether to expand input tensors. Defaults to False.
    :return: a pair ``(inputs, tensors)`` where tensors are all
        :class:`torch.Tensor` s or :class:`np.ndarray` s
        that can be broadcast together to a single data
        with given ``inputs``.
    :rtype: tuple
    """
    expand = kwargs.pop("expand", False)
    assert not kwargs
    inputs = OrderedDict()
    for x in args:
        inputs.update(x.inputs)
    tensors = [align_tensor(inputs, x, expand=expand) for x in args]
    return inputs, tensors


@to_data.register(Tensor)
def tensor_to_data(x, name_to_dim=None):
    pass


@eager.register(Scatter, Op, tuple, Number, frozenset)
def eager_scatter_number(op, subs, source, reduced_vars):
    pass


@eager.register(Scatter, Op, tuple, Tensor, frozenset)
def eager_scatter_tensor(op, subs, source, reduced_vars):
    pass


@eager.register(Binary, BinaryOp, Tensor, Number)
def eager_binary_tensor_number(op, lhs, rhs):
    pass


@eager.register(Binary, BinaryOp, Number, Tensor)
def eager_binary_number_tensor(op, lhs, rhs):
    pass


@eager.register(Binary, BinaryOp, Tensor, Tensor)
def eager_binary_tensor_tensor(op, lhs, rhs):
    pass


@eager.register(Binary, MatmulOp, Tensor, Tensor)
def eager_binary_tensor_tensor(op, lhs, rhs):
    pass


@eager.register(Unary, ReshapeOp, Tensor)
def eager_reshape_tensor(op, arg):
    pass


@eager.register(Unary, ops.ReductionOp, Tensor)
def eager_reduction_tensor(op, arg):
    pass


@eager.register(Binary, GetitemOp, Tensor, Number)
def eager_getitem_tensor_number(op, lhs, rhs):
    pass


@eager.register(Binary, GetitemOp, Tensor, Variable)
def eager_getitem_tensor_variable(op, lhs, rhs):
    pass


@eager.register(Binary, GetitemOp, Tensor, Tensor)
def eager_getitem_tensor_tensor(op, lhs, rhs):
    pass


@eager.register(Unary, ops.GetsliceOp, Tensor)
def eager_getslice_tensor(op, x):
    pass


@eager.register(
    Finitary, ops.StackOp, typing.Tuple[typing.Union[(Number, Tensor)], ...]
)
def eager_finitary_stack(op, parts):
    pass


@eager.register(Finitary, ops.CatOp, typing.Tuple[Tensor, ...])
def eager_finitary_cat(op, parts):
    pass


@eager.register(Finitary, FinitaryOp, typing.Tuple[typing.Union[(Number, Tensor)], ...])
def eager_finitary_generic_tensors(op, args):
    pass


@eager.register(Lambda, Variable, Tensor)
def eager_lambda(var, expr):
    pass


@dispatch(str, Variadic[Tensor])
def eager_stack_homogeneous(name, *parts):
    pass


@dispatch(str, str, Variadic[Tensor])
def eager_cat_homogeneous(name, part_name, *parts):
    pass


class Function(Funsor):

    def __init__(self, fn, output, args):
        assert callable(fn)
        assert not isinstance(fn, Function)
        assert isinstance(args, tuple)
        inputs = OrderedDict()
        for arg in args:
            assert isinstance(arg, Funsor)
            inputs.update(arg.inputs)
        super(Function, self).__init__(inputs, output)
        self.fn = fn
        self.args = args

    def __repr__(self):
        return "{}({}, {}, {})".format(
            type(self).__name__, _nameof(self.fn), repr(self.output), repr(self.args)
        )

    def __str__(self):
        return "{}({}, {}, {})".format(
            type(self).__name__, _nameof(self.fn), str(self.output), str(self.args)
        )


@quote.register(Function)
def _(arg, indent, out):
    pass


@eager.register(Function, object, ArrayType, tuple)
def eager_function(fn, output, args):
    pass


def _select(fn, i, *args):
    pass


def _nested_function(fn, args, output):
    if isinstance(output, ArrayType):
        return Function(fn, output, args)
    elif output.__origin__ in (tuple, Product, typing.Tuple):
        result = []
        for i, output_i in enumerate(output.__args__):
            fn_i = functools.partial(_select, fn, i)
            fn_i.__name__ = "{}_{}".format(_nameof(fn), i)
            result.append(_nested_function(fn_i, args, output_i))
        return Tuple(tuple(result))
    raise ValueError("Invalid output: {}".format(output))


class _Memoized(object):
    def __init__(self, fn):
        self.fn = fn
        self._cache = None

    def __call__(self, *args):
        if self._cache is not None:
            old_args, old_result = self._cache
            if all(x is y for x, y in zip(args, old_args)):
                return old_result
        result = self.fn(*args)
        self._cache = args, result
        return result

    @property
    def __name__(self):
        return _nameof(self.fn)

    @property
    def __annotations__(self):
        return self.fn.__annotations__


def _function(inputs, output, fn):
    if is_nn_module(fn):
        names = getargspec(fn.forward)[0][1:]
    else:
        names = getargspec(fn)[0]
    if isinstance(inputs, dict):
        args = tuple(Variable(name, inputs[name]) for name in names if name in inputs)
    else:
        args = tuple(Variable(name, domain) for (name, domain) in zip(names, inputs))
    assert len(args) == len(inputs)
    if not isinstance(output, ArrayType):
        assert output.__origin__ in (tuple, Product, typing.Tuple)
        fn = _Memoized(fn)
    return _nested_function(fn, args, output)


def _tuple_to_Tuple(tp):
    if isinstance(tp, tuple):
        warnings.warn(
            "tuple types like (Real, Reals[2]) are deprecated, "
            "use Tuple[Real, Reals[2]] instead",
            DeprecationWarning,
        )
        tp = tuple(map(_tuple_to_Tuple, tp))
        return typing.Tuple[tp]
    return tp


def function(*signature):
    r"""
    Decorator to wrap a PyTorch/NumPy function, using either type hints or
    explicit type annotations.

    Example::

        # Using type hints:
        @funsor.tensor.function
        def matmul(x: Reals[3, 4], y: Reals[4, 5]) -> Reals[3, 5]:
            return torch.matmul(x, y)

        # Using explicit type annotations:
        @funsor.tensor.function(Reals[3, 4], Reals[4, 5], Reals[3, 5])
        def matmul(x, y):
            return torch.matmul(x, y)

        @funsor.tensor.function(Reals[10], Reals[10, 10], Reals[10], Real)
        def mvn_log_prob(loc, scale_tril, x):
            d = torch.distributions.MultivariateNormal(loc, scale_tril)
            return d.log_prob(x)

    To support functions that output nested tuples of tensors, specify a nested
    :py:class:`~typing.Tuple` of output types, for example::

        @funsor.tensor.function
        def max_and_argmax(x: Reals[8]) -> Tuple[Real, Bint[8]]:
            return torch.max(x, dim=-1)

    :param \*signature: A sequence if input domains followed by a final output
        domain or nested tuple of output domains.
    """
    assert signature
    if len(signature) == 1:
        fn = signature[0]
        if callable(fn) and not isinstance(fn, ArrayType):
            inputs = typing.get_type_hints(as_callable(fn))
            output = inputs.pop("return")
            assert all(isinstance(d, ArrayType) for d in inputs.values())
            assert isinstance(output, (ArrayType, tuple)) or output.__origin__ in (
                tuple,
                Product,
                typing.Tuple,
            )
            return _function(inputs, output, fn)
    inputs, output = signature[:-1], signature[-1]
    output = _tuple_to_Tuple(output)
    assert all(isinstance(d, ArrayType) for d in inputs)
    assert isinstance(output, (ArrayType, tuple)) or output.__origin__ in (
        tuple,
        Product,
        typing.Tuple,
    )
    return functools.partial(_function, inputs, output)


def Einsum(equation, *operands):
    """
    Wrapper around :func:`torch.einsum` or :func:`np.einsum` to operate on real-valued Funsors.

    Note this operates only on the ``output`` tensor. To perform sum-product
    contractions on named dimensions, instead use ``+`` and
    :class:`~funsor.terms.Reduce`.

    :param str equation: An :func:`torch.einsum` or :func:`np.einsum` equation.
    :param tuple operands: A tuple of input funsors.
    """
    return ops.einsum(operands, equation)


@eager.register(Finitary, ops.EinsumOp, typing.Tuple[Tensor, ...])
def eager_einsum(op, operands):
    pass


def tensordot(x, y, dims):
    """
    Wrapper around :func:`torch.tensordot` or :func:`np.tensordot`
    to operate on real-valued Funsors.

    Note this operates only on the ``output`` tensor. To perform sum-product
    contractions on named dimensions, instead use ``+`` and
    :class:`~funsor.terms.Reduce`.

    Arguments should satisfy::

        len(x.shape) >= dims
        len(y.shape) >= dims
        dims == 0 or x.shape[-dims:] == y.shape[:dims]

    :param Funsor x: A left hand argument.
    :param Funsor y: A y hand argument.
    :param int dims: The number of dimension of overlap of output shape.
    :rtype: Funsor
    """
    assert dims >= 0
    assert len(x.shape) >= dims
    assert len(y.shape) >= dims
    assert dims == 0 or x.shape[-dims:] == y.shape[:dims]
    x_start, x_end = 0, len(x.output.shape)
    y_start = x_end - dims
    y_end = y_start + len(y.output.shape)
    symbols = "abcdefghijklmnopqrstuvwxyz"
    equation = "{},{}->{}".format(
        symbols[x_start:x_end],
        symbols[y_start:y_end],
        symbols[x_start:y_start] + symbols[x_end:y_end],
    )
    return Einsum(equation, x, y)


REDUCE_OP_TO_NUMERIC = {
    ops.add: ops.sum,
    ops.mul: ops.prod,
    ops.and_: ops.all,
    ops.or_: ops.any,
    ops.logaddexp: ops.logsumexp,
    ops.sample: ops.logsumexp,
    ops.min: ops.amin,
    ops.max: ops.amax,
}


__all__ = [
    "Einsum",
    "Function",
    "REDUCE_OP_TO_NUMERIC",
    "Tensor",
    "align_tensor",
    "align_tensors",
    "function",
    "ignore_jit_warnings",
    "tensordot",
]
