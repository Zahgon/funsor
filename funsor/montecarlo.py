
import functools
from collections import OrderedDict

from funsor.cnf import Contraction
from funsor.delta import Delta
from funsor.gaussian import Gaussian
from funsor.integrate import Integrate
from funsor.interpretations import StatefulInterpretation
from funsor.tensor import Tensor
from funsor.terms import Approximate, Funsor, Number, Subs, Unary
from funsor.util import get_backend

from . import ops


class MonteCarlo(StatefulInterpretation):

    def __init__(self, *, rng_key=None, **sample_inputs):
        super().__init__("monte_carlo")
        self.rng_key = rng_key
        self.sample_inputs = OrderedDict(sample_inputs)


@MonteCarlo.register(Integrate, Funsor, Funsor, frozenset)
def monte_carlo_integrate(state, log_measure, integrand, reduced_vars):
    pass


@MonteCarlo.register(Approximate, ops.LogaddexpOp, Funsor, Funsor, frozenset)
def monte_carlo_approximate(state, op, model, guide, approx_vars):
    pass


@functools.singledispatch
def extract_samples(discrete_density):
    """
    Extract sample values out of a funsor Delta, possibly scaled by Tensors.
    This is useful for extracting sample tensors from a Monte Carlo
    computation.
    """
    raise ValueError(
        f"Could not extract support from {type(discrete_density).__name__}"
    )


@extract_samples.register(Delta)
def _extract_samples_delta(discrete_density):
    pass


@extract_samples.register(Contraction)
def _extract_samples_contraction(discrete_density):
    pass


@extract_samples.register(Subs)
@extract_samples.register(Number)
@extract_samples.register(Tensor)
@extract_samples.register(Gaussian)
@extract_samples.register(Unary)
def _extract_samples_scale(discrete_density):
    pass


__all__ = [
    "MonteCarlo",
]
