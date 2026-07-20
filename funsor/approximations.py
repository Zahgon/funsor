
from collections import OrderedDict
from functools import reduce, singledispatch

from funsor.cnf import Contraction, GaussianMixture
from funsor.delta import Delta
from funsor.gaussian import Gaussian, _compute_offsets
from funsor.instrument import debug_logged
from funsor.integrate import Integrate
from funsor.interpretations import DispatchedInterpretation
from funsor.tensor import Tensor, align_tensor
from funsor.terms import Approximate, Funsor
from funsor.typing import deep_isinstance

from . import ops

argmax_approximate = DispatchedInterpretation("argmax_approximate")
"""
Point-approximate at the argmax of the provided guide.
"""


@argmax_approximate.register(Approximate, ops.MaxOp, Funsor, Funsor, frozenset)
@argmax_approximate.register(Approximate, ops.LogaddexpOp, Funsor, Funsor, frozenset)
def argmax_approximate_logaddexp(op, model, guide, approx_vars):
    pass


mean_approximate = DispatchedInterpretation("mean_approximate")
"""
Point-approximate at the mean of the provided guide.
"""


@mean_approximate.register(Approximate, ops.LogaddexpOp, Funsor, Funsor, frozenset)
def mean_approximate_logaddexp(op, model, guide, approx_vars):
    pass


laplace_approximate = DispatchedInterpretation("laplace_approximate")
"""
Gaussian approximate using the value and Hessian of the model, evaluated at the
mode of the guide.
"""


@laplace_approximate.register(Approximate, ops.LogaddexpOp, Funsor, Funsor, frozenset)
def laplace_approximate_logaddexp(op, model, guide, approx_vars):
    pass




@singledispatch
def compute_hessian(model, approx_vars):
    raise NotImplementedError


@singledispatch
def compute_argmax(model, approx_vars):
    pass


@compute_argmax.register(Tensor)
@debug_logged
def compute_argmax_tensor(model, approx_vars):
    pass


@compute_argmax.register(Gaussian)
@debug_logged
def compute_argmax_gaussian(model, approx_vars):
    pass


@compute_argmax.register(GaussianMixture)
@debug_logged
def compute_argmax_gaussian_mixture(model, approx_vars):
    pass


@compute_argmax.register(Contraction[ops.NullOp, ops.AddOp, frozenset, tuple])
def compute_argmax_contract(model, approx_vars):
    pass


__all__ = [
    "argmax_approximate",
    "compute_argmax",
    "laplace_approximate",
    "mean_approximate",
]
