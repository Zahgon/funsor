
from collections import OrderedDict
from functools import reduce

from funsor.domains import Domain, Real
from funsor.instrument import debug_logged
from funsor.ops import AddOp, SubOp, TransformOp
from funsor.registry import KeyedRegistry
from funsor.terms import (
    Align,
    Binary,
    Funsor,
    FunsorMeta,
    Independent,
    Lambda,
    Number,
    Unary,
    Variable,
    eager,
    to_funsor,
)
from funsor.util import get_default_dtype

from . import ops


def solve(expr, value):
    """
    Tries to solve for free inputs of an ``expr`` such that ``expr == value``,
    and computes the log-abs-det-Jacobian of the resulting substitution.

    :param Funsor expr: An expression with a free variable.
    :param Funsor value: A target value.
    :return: A tuple ``(name, point, log_abs_det_jacobian)``
    :rtype: tuple
    :raises: ValueError
    """
    assert isinstance(expr, Funsor)
    assert isinstance(value, Funsor)
    result = solve.dispatch(type(expr), *(expr._ast_values + (value,)))
    if result is None:
        raise ValueError("Cannot substitute into a Delta: {}".format(value))
    return result


_solve = KeyedRegistry(lambda *args: None)
solve.dispatch = _solve.__call__
solve.register = _solve.register


@solve.register(Variable, str, Domain, Funsor)
@debug_logged
def solve_variable(name, output, y):
    pass


@solve.register(Unary, TransformOp, Funsor, Funsor)
@debug_logged
def solve_unary(op, arg, y):
    pass


class DeltaMeta(FunsorMeta):

    def __call__(cls, *args):
        if len(args) > 1:
            assert len(args) == 2 or len(args) == 3
            assert isinstance(args[0], str) and isinstance(args[1], Funsor)
            args = args + (Number(0.0),) if len(args) == 2 else args
            args = (((args[0], (to_funsor(args[1]), to_funsor(args[2]))),),)
        assert isinstance(args[0], tuple)
        return super().__call__(args[0])


class Delta(Funsor, metaclass=DeltaMeta):

    def __init__(self, terms):
        assert isinstance(terms, tuple) and len(terms) > 0
        inputs = OrderedDict()
        for name, (point, log_density) in terms:
            assert isinstance(name, str)
            assert isinstance(point, Funsor)
            assert isinstance(log_density, Funsor)
            assert log_density.output == Real
            assert name not in inputs
            assert name not in point.inputs
            inputs.update({name: point.output})
            inputs.update(point.inputs)

        output = Real
        fresh = frozenset(name for name, term in terms)
        bound = {}
        super(Delta, self).__init__(inputs, output, fresh, bound)
        self.terms = terms

    def align(self, names):
        assert isinstance(names, tuple)
        assert all(name in self.fresh for name in names)
        if not names or names == tuple(n for n, p in self.terms):
            return self

        new_terms = tuple(sorted(self.terms, key=lambda t: names.index(t[0])))
        return Delta(new_terms)

    def eager_subs(self, subs):
        pass

    def eager_reduce(self, op, reduced_vars):
        pass

    def _sample(self, sampled_vars, sample_inputs, rng_key):
        pass


@eager.register(Binary, AddOp, Delta, Delta)
def eager_add_multidelta(op, lhs, rhs):
    pass


@eager.register(Binary, (AddOp, SubOp), Delta, (Funsor, Align))
def eager_add_delta_funsor(op, lhs, rhs):
    pass


@eager.register(Binary, AddOp, (Funsor, Align), Delta)
def eager_add_funsor_delta(op, lhs, rhs):
    pass


@eager.register(Independent, Delta, str, str, str)
def eager_independent_delta(delta, reals_var, bint_var, diag_var):
    pass


__all__ = [
    "Delta",
    "solve",
]
