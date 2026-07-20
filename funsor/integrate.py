
from collections import OrderedDict
from functools import reduce
from typing import Tuple, Union

import funsor.ops as ops
from funsor.cnf import Contraction, GaussianMixture
from funsor.constant import Constant
from funsor.delta import Delta
from funsor.gaussian import Gaussian, _norm2, _vm, align_gaussian
from funsor.interpretations import eager, normalize
from funsor.tensor import Tensor
from funsor.terms import (
    Funsor,
    FunsorMeta,
    Number,
    Subs,
    Unary,
    Variable,
    _convert_reduced_vars,
    substitute,
    to_funsor,
)


class IntegrateMeta(FunsorMeta):

    def __call__(cls, log_measure, integrand, reduced_vars):
        inputs = log_measure.inputs.copy()
        inputs.update(integrand.inputs)
        reduced_vars = _convert_reduced_vars(reduced_vars, inputs)
        return super().__call__(log_measure, integrand, reduced_vars)


class Integrate(Funsor, metaclass=IntegrateMeta):

    def __init__(self, log_measure, integrand, reduced_vars):
        assert isinstance(log_measure, Funsor)
        assert isinstance(integrand, Funsor)
        assert isinstance(reduced_vars, frozenset)
        assert all(isinstance(v, Variable) for v in reduced_vars)
        reduced_names = frozenset(v.name for v in reduced_vars)
        inputs = OrderedDict(
            (k, d)
            for term in (log_measure, integrand)
            for (k, d) in term.inputs.items()
            if k not in reduced_names
        )
        output = integrand.output
        fresh = frozenset()
        bound = {v.name: v.output for v in reduced_vars}
        super(Integrate, self).__init__(inputs, output, fresh, bound)
        self.log_measure = log_measure
        self.integrand = integrand
        self.reduced_vars = reduced_vars

    def _alpha_convert(self, alpha_subs):
        assert set(self.bound).issuperset(alpha_subs)
        reduced_vars = frozenset(
            Variable(alpha_subs.get(v.name, v.name), v.output)
            for v in self.reduced_vars
        )
        alpha_subs = {
            k: to_funsor(
                v, self.integrand.inputs.get(k, self.log_measure.inputs.get(k))
            )
            for k, v in alpha_subs.items()
        }
        log_measure = substitute(self.log_measure, alpha_subs)
        integrand = substitute(self.integrand, alpha_subs)
        return log_measure, integrand, reduced_vars


@normalize.register(Integrate, Funsor, Funsor, frozenset)
def normalize_integrate(log_measure, integrand, reduced_vars):
    pass


@normalize.register(
    Integrate,
    Contraction[Union[ops.NullOp, ops.LogaddexpOp], ops.AddOp, frozenset, tuple],
    Funsor,
    frozenset,
)
def normalize_integrate_contraction(log_measure, integrand, reduced_vars):
    pass


EagerConstant = Constant[
    Tuple,
    Union[
        Variable,
        Delta,
        Gaussian,
        Unary[ops.NegOp, Gaussian],
        Number,
        Tensor,
        GaussianMixture,
    ],
]


@eager.register(
    Contraction,
    ops.AddOp,
    ops.MulOp,
    frozenset,
    Unary[ops.ExpOp, Union[GaussianMixture, Delta, Gaussian, Number, Tensor]],
    (
        Variable,
        Delta,
        Gaussian,
        Unary[ops.NegOp, Gaussian],
        Number,
        Tensor,
        GaussianMixture,
        EagerConstant,
    ),
)
def eager_contraction_binary_to_integrate(red_op, bin_op, reduced_vars, lhs, rhs):
    pass


@eager.register(Integrate, GaussianMixture, Funsor, frozenset)
def eager_integrate_gaussianmixture(log_measure, integrand, reduced_vars):
    pass




@eager.register(Integrate, Delta, Funsor, frozenset)
def eager_integrate(delta, integrand, reduced_vars):
    pass




@eager.register(Integrate, Gaussian, Variable, frozenset)
def eager_integrate_gaussian_variable(log_measure, integrand, reduced_vars):
    pass


@eager.register(Integrate, Gaussian, Gaussian, frozenset)
def eager_integrate_gaussian_gaussian(log_measure, integrand, reduced_vars):
    pass


@eager.register(Integrate, Gaussian, Unary[ops.NegOp, Gaussian], frozenset)
def eager_integrate_neg_gaussian(log_measure, integrand, reduced_vars):
    pass


@eager.register(
    Integrate,
    Gaussian,
    Contraction[
        ops.NullOp,
        ops.AddOp,
        frozenset,
        Tuple[Union[Gaussian, Unary[ops.NegOp, Gaussian]], ...],
    ],
    frozenset,
)
def eager_distribute_integrate(log_measure, integrand, reduced_vars):
    pass


__all__ = [
    "Integrate",
]
