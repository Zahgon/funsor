
from collections import OrderedDict

from . import ops
from .cnf import Contraction, GaussianMixture
from .domains import Reals
from .gaussian import Gaussian
from .interpretations import StatefulInterpretation
from .terms import Approximate, Funsor, Subs, Variable


class Precondition(StatefulInterpretation):

    def __init__(self, aux_name="aux"):
        super().__init__("precondition")
        self.aux_name = aux_name
        self.sample_inputs = OrderedDict()
        self.sample_vars = set()

    def combine_subs(self):
        """
        Method to create a combining substitution after preconditioning is
        complete. The returned substitution replaces per-factor auxiliary
        variables with slices into a single combined auxiliary variable.

        :returns: A substitution indexing each factor-wise auxiliary variable
            into a single global auxiliary variable.
        :rtype: dict
        """
        total_size = sum(v.num_elements for v in self.sample_inputs.values())
        aux = Variable(self.aux_name, Reals[total_size])
        subs = {}
        start = 0
        for k, v in self.sample_inputs.items():
            stop = start + v.num_elements
            subs[k] = aux[start:stop].reshape(v.shape)
            start = stop
        return subs


@Precondition.register(Approximate, ops.LogaddexpOp, Funsor, Funsor, frozenset)
def precondition_approximate_todo(state, op, model, guide, approx_vars):
    pass


@Precondition.register(
    Approximate,
    ops.LogaddexpOp,
    Funsor,
    Contraction[ops.NullOp, ops.AddOp, frozenset, tuple],
    frozenset,
)
def precondition_approximate_contraction(state, op, model, guide, approx_vars):
    pass


@Precondition.register(Approximate, ops.LogaddexpOp, Funsor, GaussianMixture, frozenset)
def precondition_approximate_gaussian_mixture(state, op, model, guide, approx_vars):
    pass


@Precondition.register(Approximate, ops.LogaddexpOp, Funsor, Gaussian, frozenset)
@Precondition.register(
    Approximate, ops.LogaddexpOp, Funsor, Subs[Gaussian, tuple], frozenset
)
def precondition_approximate_gaussian(state, op, model, guide, approx_vars):
    pass


__all__ = [
    "Precondition",
]
