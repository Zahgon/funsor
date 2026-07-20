
from collections import OrderedDict
from functools import reduce, singledispatch

import opt_einsum

from funsor.domains import Bint
from funsor.interpreter import gensym
from funsor.tensor import Tensor, get_default_prototype
from funsor.terms import Binary, Finitary, Funsor, Lambda, Reduce, Unary, Variable

from . import ops


def is_affine(fn):
    """
    A sound but incomplete test to determine whether a funsor is affine with
    respect to all of its real inputs.

    :param Funsor fn: A funsor.
    :rtype: bool
    """
    return affine_inputs(fn) == _real_inputs(fn)


def _real_inputs(fn):
    return frozenset(k for k, d in fn.inputs.items() if d.dtype == "real")


def affine_inputs(fn):
    """
    Returns a [sound sub]set of real inputs of ``fn``
    wrt which ``fn`` is known to be affine.

    :param Funsor fn: A funsor.
    :return: A set of input names wrt which ``fn`` is affine.
    :rtype: frozenset
    """
    result = getattr(fn, "_affine_inputs", None)
    if result is None:
        result = fn._affine_inputs = _affine_inputs(fn)
    return result


@singledispatch
def _affine_inputs(fn):
    assert isinstance(fn, Funsor)
    return frozenset()


affine_inputs.register = _affine_inputs.register


@affine_inputs.register(Variable)
def _(fn):
    pass


@affine_inputs.register(Unary)
def _(fn):
    pass


@affine_inputs.register(Binary)
def _(fn):
    pass


@affine_inputs.register(Reduce)
def _(fn):
    pass


@affine_inputs.register(Finitary[ops.EinsumOp, tuple])
def _(fn):
    pass


def extract_affine(fn):
    """
    Extracts an affine representation of a funsor, satisfying::

        x = ...
        const, coeffs = extract_affine(x)
        y = sum(Einsum(eqn, coeff, Variable(var, coeff.output))
                for var, (coeff, eqn) in coeffs.items())
        assert_close(y, x)
        assert frozenset(coeffs) == affine_inputs(x)

    The ``coeffs`` will have one key per input wrt which ``fn`` is known to be
    affine (via :func:`affine_inputs` ), and ``const`` and ``coeffs.values``
    will all be constant wrt these inputs.

    The affine approximation is computed by ev evaluating ``fn`` at
    zero and each basis vector. To improve performance, users may want to run
    under the :func:`~funsor.interpretations.Memoize` interpretation.

    :param Funsor fn: A funsor that is affine wrt the (add,mul) semiring in
        some subset of its inputs.
    :return: A pair ``(const, coeffs)`` where const is a funsor with no real
        inputs and ``coeffs`` is an OrderedDict mapping input name to a
        ``(coefficient, eqn)`` pair in einsum form.
    :rtype: tuple
    """
    prototype = get_default_prototype()
    inputs = affine_inputs(fn)
    inputs = OrderedDict((k, v) for k, v in fn.inputs.items() if k in inputs)
    zeros = {k: Tensor(ops.new_zeros(prototype, v.shape)) for k, v in inputs.items()}
    const = fn(**zeros)

    name = gensym("probe")
    coeffs = OrderedDict()
    for k, v in inputs.items():
        dim = v.num_elements
        var = Variable(name, Bint[dim])
        subs = zeros.copy()
        subs[k] = Tensor(ops.new_eye(prototype, (dim,)).reshape((dim,) + v.shape))[var]
        coeff = Lambda(var, fn(**subs) - const).reshape(v.shape + const.shape)
        inputs1 = "".join(map(opt_einsum.get_symbol, range(len(coeff.shape))))
        inputs2 = inputs1[: len(v.shape)]
        output = inputs1[len(v.shape) :]
        eqn = "{},{}->{}".format(inputs1, inputs2, output)
        coeffs[k] = coeff, eqn
    return const, coeffs


__all__ = [
    "affine_inputs",
    "extract_affine",
    "is_affine",
]
