
from collections import OrderedDict
from functools import reduce

import funsor.ops as ops
from funsor.tensor import Tensor
from funsor.terms import (
    Binary,
    Funsor,
    FunsorMeta,
    Number,
    Reduce,
    Unary,
    Variable,
    eager,
)


class ConstantMeta(FunsorMeta):

    def __call__(cls, const_inputs, arg):
        if isinstance(const_inputs, dict):
            const_inputs = tuple(const_inputs.items())
        return super(ConstantMeta, cls).__call__(const_inputs, arg)


class Constant(Funsor, metaclass=ConstantMeta):

    def __init__(self, const_inputs, arg):
        assert isinstance(arg, Funsor)
        assert isinstance(const_inputs, tuple)
        const_inputs = OrderedDict(const_inputs)
        assert set(const_inputs).isdisjoint(arg.inputs)
        inputs = const_inputs.copy()
        inputs.update(arg.inputs)
        output = arg.output
        fresh = frozenset(const_inputs)
        bound = {}
        super(Constant, self).__init__(inputs, output, fresh, bound)
        self.arg = arg
        self.const_vars = frozenset(Variable(k, v) for k, v in const_inputs.items())
        self.const_inputs = const_inputs

    def eager_subs(self, subs):
        pass

    def eager_reduce(self, op, reduced_vars):
        pass

    def align(self, names):
        assert isinstance(names, tuple)
        assert all(name in self.inputs for name in names)
        if not names or names == tuple(self.inputs):
            return self

        const_names = names[: len(self.const_inputs)]
        arg_names = names[len(self.const_inputs) :]
        assert frozenset(self.const_inputs) == frozenset(const_names)
        const_inputs = OrderedDict((name, self.inputs[name]) for name in const_names)
        return Constant(const_inputs, self.arg.align(arg_names))

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

        assert isinstance(self.arg, Tensor)
        return self.arg.materialize(x)


@eager.register(Reduce, ops.AddOp, Constant, frozenset)
@eager.register(Reduce, ops.MulOp, Constant, frozenset)
@eager.register(Reduce, ops.LogaddexpOp, Constant, frozenset)
def eager_reduce_add(op, arg, reduced_vars):
    pass


@eager.register(Binary, ops.BinaryOp, Constant, Constant)
def eager_binary_constant_constant(op, lhs, rhs):
    pass


@eager.register(Binary, ops.BinaryOp, Constant, (Number, Tensor))
def eager_binary_constant_tensor(op, lhs, rhs):
    pass


@eager.register(Binary, ops.BinaryOp, (Number, Tensor), Constant)
def eager_binary_tensor_constant(op, lhs, rhs):
    pass


@eager.register(Unary, ops.UnaryOp, Constant)
def eager_unary(op, arg):
    pass
