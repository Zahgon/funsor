
from functools import reduce

from multipledispatch.variadic import Variadic

from funsor.cnf import Contraction
from funsor.integrate import Integrate
from funsor.interpretations import StatefulInterpretation
from funsor.ops import AddOp, LogaddexpOp, NullOp
from funsor.terms import Funsor, Reduce

from . import ops


class Elbo(StatefulInterpretation):

    def __init__(self, guide, approx_vars):
        super().__init__("elbo")
        self.guide = guide
        self.approx_vars = approx_vars


@Elbo.register(Contraction, (LogaddexpOp, NullOp), (AddOp, NullOp), frozenset, tuple)
def elbo_contract(state, sum_op, prod_op, reduced_vars, terms):
    pass


@Elbo.register(
    Contraction, (LogaddexpOp, NullOp), (AddOp, NullOp), frozenset, Variadic[Funsor]
)
def elbo_contract_variadic(state, sum_op, prod_op, reduced_vars, *terms):
    pass


@Elbo.register(Reduce, LogaddexpOp, Funsor, frozenset)
def elbo_reduce(state, sum_op, arg, reduced_vars):
    pass


__all__ = [
    "Elbo",
]
