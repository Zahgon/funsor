
import collections

from opt_einsum.paths import greedy

import funsor.interpreter as interpreter
from funsor.cnf import Contraction
from funsor.interpretations import (
    DispatchedInterpretation,
    PrioritizedInterpretation,
    eager,
    lazy,
    normalize_base,
)
from funsor.interpreter import get_interpretation
from funsor.ops import DISTRIBUTIVE_OPS, AssociativeOp
from funsor.terms import Funsor
from funsor.typing import Variadic

from . import ops

unfold_base = DispatchedInterpretation()
unfold = PrioritizedInterpretation(unfold_base, normalize_base, lazy)


@unfold.register(Contraction, AssociativeOp, AssociativeOp, frozenset, tuple)
def unfold_contraction_generic_tuple(red_op, bin_op, reduced_vars, terms):
    pass


@unfold.register(Contraction, AssociativeOp, AssociativeOp, frozenset, Variadic[Funsor])
def unfold_contraction_variadic(r, b, v, *ts):
    pass


optimize_base = DispatchedInterpretation()
optimize = PrioritizedInterpretation(optimize_base, eager)


REAL_SIZE = 3  # the "size" of a real-valued dimension passed to the path optimizer


@optimize.register(
    Contraction, AssociativeOp, AssociativeOp, frozenset, Variadic[Funsor]
)
def optimize_contraction_variadic(r, b, v, *ts):
    pass


@optimize.register(Contraction, AssociativeOp, AssociativeOp, frozenset, Funsor, Funsor)
@optimize.register(Contraction, AssociativeOp, AssociativeOp, frozenset, Funsor)
def eager_contract_base(red_op, bin_op, reduced_vars, *terms):
    pass


@optimize.register(Contraction, AssociativeOp, AssociativeOp, frozenset, tuple)
def optimize_contract_finitary_funsor(red_op, bin_op, reduced_vars, terms):
    pass


def apply_optimizer(x):
    with unfold:
        expr = interpreter.reinterpret(x)

    with PrioritizedInterpretation(optimize_base, get_interpretation()):
        return interpreter.reinterpret(expr)
