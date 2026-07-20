
import functools
import numbers
from typing import Tuple, Union

import pyro.distributions as dist
import pyro.distributions.testing.fakes as fakes
import torch
from pyro.distributions.torch_distribution import (
    ExpandedDistribution,
    MaskedDistribution,
)

import funsor.ops as ops
from funsor.cnf import Contraction
from funsor.constant import Constant
from funsor.distribution import (  # noqa: F401
    FUNSOR_DIST_NAMES,
    Bernoulli,
    LogNormal,
    backenddist_to_funsor,
    eager_beta,
    eager_beta_bernoulli,
    eager_binomial,
    eager_categorical_funsor,
    eager_categorical_tensor,
    eager_delta_funsor_funsor,
    eager_delta_funsor_variable,
    eager_delta_tensor,
    eager_delta_variable_variable,
    eager_dirichlet_categorical,
    eager_dirichlet_multinomial,
    eager_dirichlet_posterior,
    eager_gamma_gamma,
    eager_gamma_poisson,
    eager_multinomial,
    eager_mvn,
    eager_normal,
    eager_plate_multinomial,
    expandeddist_to_funsor,
    indepdist_to_funsor,
    make_dist,
    maskeddist_to_funsor,
    transformeddist_to_funsor,
)
from funsor.domains import Real, Reals
from funsor.interpretations import eager
from funsor.tensor import Tensor
from funsor.terms import Binary, Funsor, Reduce, Unary, Variable, to_data, to_funsor
from funsor.util import methodof

__all__ = list(x[0] for x in FUNSOR_DIST_NAMES)




class _PyroWrapper_BernoulliProbs(dist.Bernoulli):
    def __init__(self, probs, validate_args=None):
        return super().__init__(probs=probs, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(_PyroWrapper_BernoulliProbs, _instance)
        return super().expand(batch_shape, _instance=new)


class _PyroWrapper_BernoulliLogits(dist.Bernoulli):
    def __init__(self, logits, validate_args=None):
        return super().__init__(logits=logits, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(_PyroWrapper_BernoulliLogits, _instance)
        return super().expand(batch_shape, _instance=new)


class _PyroWrapper_CategoricalLogits(dist.Categorical):
    def __init__(self, logits, validate_args=None):
        return super().__init__(logits=logits, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(_PyroWrapper_CategoricalLogits, _instance)
        return super().expand(batch_shape, _instance=new)


def _get_pyro_dist(dist_name):
    pass


PYRO_DIST_NAMES = FUNSOR_DIST_NAMES + [
    ("ContinuousBernoulli", ("logits",)),
    ("FisherSnedecor", ()),
    ("NegativeBinomial", ("total_count", "probs")),
    ("OneHotCategorical", ("probs",)),
    ("RelaxedBernoulli", ("temperature", "logits")),
    ("Weibull", ()),
]


for dist_name, param_names in PYRO_DIST_NAMES:
    locals()[dist_name] = make_dist(_get_pyro_dist(dist_name), param_names=param_names)


@methodof(Delta)  # noqa: F821
@staticmethod
def _infer_value_domain(**kwargs):
    return kwargs["v"]


@methodof(Categorical)  # noqa: F821
@methodof(CategoricalLogits)  # noqa: F821
@classmethod
def _infer_value_dtype(cls, domains):
    if "logits" in domains:
        return domains["logits"].shape[-1]
    if "probs" in domains:
        return domains["probs"].shape[-1]
    raise ValueError


@methodof(Binomial)  # noqa: F821
@methodof(Multinomial)  # noqa: F821
@methodof(DirichletMultinomial)  # noqa: F821
@classmethod
def _infer_value_dtype(cls, domains):
    return "real"


@methodof(Delta)  # noqa: F821
@staticmethod
@functools.lru_cache(maxsize=5000)
def _infer_param_domain(name, raw_shape):
    if name == "v":
        return Reals[raw_shape]
    elif name == "log_density":
        return Real
    else:
        raise ValueError(name)


@methodof(Dirichlet)  # noqa: F821
@methodof(NonreparameterizedDirichlet)  # noqa: F821
@staticmethod
@functools.lru_cache(maxsize=5000)
def _infer_param_domain(name, raw_shape):
    assert name == "concentration"
    return Reals[raw_shape[-1]]


@methodof(DirichletMultinomial)  # noqa: F821
@classmethod
@functools.lru_cache(maxsize=5000)
def _infer_param_domain(cls, name, raw_shape):
    if name == "concentration":
        return Reals[raw_shape[-1]]
    assert name == "total_count"
    return Real


@methodof(LowRankMultivariateNormal)  # noqa: F821
@classmethod
@functools.lru_cache(maxsize=5000)
def _infer_param_domain(cls, name, raw_shape):
    if name == "loc":
        return Reals[raw_shape[-1]]
    elif name == "cov_factor":
        return Reals[raw_shape[-2:]]
    elif name == "cov_diag":
        return Reals[raw_shape[-1]]
    raise ValueError(f"{name} invalid param for {cls}")


@methodof(RelaxedBernoulli)  # noqa: F821
@classmethod
@functools.lru_cache(maxsize=5000)
def _infer_param_domain(cls, name, raw_shape):
    if name == "temperature":
        return Real
    return Real




@to_data.register(Multinomial)  # noqa: F821
def multinomial_to_data(funsor_dist, name_to_dim=None):
    pass


@to_data.register(Delta)  # noqa: F821
def deltadist_to_data(funsor_dist, name_to_dim=None):
    pass


@functools.singledispatch
def op_to_torch_transform(op, name_to_dim=None):
    raise NotImplementedError("cannot convert {} to a Transform".format(op))


@op_to_torch_transform.register(ops.TransformOp)
def transform_to_torch_transform(op, name_to_dim=None):
    raise NotImplementedError("{} is not a currently supported transform".format(op))


@op_to_torch_transform.register(ops.WrappedTransformOp)
def transform_to_torch_transform(op, name_to_dim=None):
    pass


@op_to_torch_transform.register(ops.ExpOp)
def exp_to_torch_transform(op, name_to_dim=None):
    pass


@op_to_torch_transform.register(ops.LogOp)
def log_to_torch_transform(op, name_to_dim=None):
    pass


@op_to_torch_transform.register(ops.SigmoidOp)
def sigmoid_to_torch_transform(op, name_to_dim=None):
    pass


@op_to_torch_transform.register(ops.TanhOp)
def tanh_to_torch_transform(op, name_to_dim=None):
    pass


@op_to_torch_transform.register(ops.AtanhOp)
def atanh_to_torch_transform(op, name_to_dim=None):
    pass


@to_data.register(Unary[ops.TransformOp, Union[Unary, Variable]])
def transform_to_data(expr, name_to_dim=None):
    pass




@to_funsor.register(torch.distributions.Transform)
def transform_to_funsor(tfm, output=None, dim_to_name=None, real_inputs=None):
    pass


@to_funsor.register(torch.distributions.transforms.ExpTransform)
def exptransform_to_funsor(tfm, output=None, dim_to_name=None, real_inputs=None):
    pass


@to_funsor.register(torch.distributions.transforms.TanhTransform)
def exptransform_to_funsor(tfm, output=None, dim_to_name=None, real_inputs=None):
    pass


@to_funsor.register(torch.distributions.transforms.SigmoidTransform)
def exptransform_to_funsor(tfm, output=None, dim_to_name=None, real_inputs=None):
    pass


@to_funsor.register(torch.distributions.transforms._InverseTransform)
def inversetransform_to_funsor(tfm, output=None, dim_to_name=None, real_inputs=None):
    pass


@to_funsor.register(torch.distributions.transforms.ComposeTransform)
def composetransform_to_funsor(tfm, output=None, dim_to_name=None, real_inputs=None):
    pass


to_funsor.register(ExpandedDistribution)(expandeddist_to_funsor)
to_funsor.register(torch.distributions.Independent)(indepdist_to_funsor)
to_funsor.register(MaskedDistribution)(maskeddist_to_funsor)
to_funsor.register(torch.distributions.TransformedDistribution)(
    transformeddist_to_funsor
)


@to_funsor.register(torch.distributions.Bernoulli)
def bernoulli_to_funsor(pyro_dist, output=None, dim_to_name=None):
    pass


@to_funsor.register(dist.Delta)  # Delta **distribution**
def deltadist_to_funsor(pyro_dist, output=None, dim_to_name=None):
    pass


JointDirichletMultinomial = Contraction[
    Union[ops.LogaddexpOp, ops.NullOp],
    ops.AddOp,
    frozenset,
    Tuple[Dirichlet, Multinomial],  # noqa: F821
]


eager.register(Beta, Funsor, Funsor, Funsor)(eager_beta)  # noqa: F821)
eager.register(Binomial, Funsor, Funsor, Funsor)(eager_binomial)  # noqa: F821
eager.register(Multinomial, Tensor, Tensor, Tensor)(eager_multinomial)  # noqa: F821)
eager.register(Categorical, Funsor, Tensor)(eager_categorical_funsor)  # noqa: F821)
eager.register(Categorical, Tensor, Variable)(eager_categorical_tensor)  # noqa: F821)
eager.register(Categorical, Constant[Tuple, Tensor], Variable)(
    eager_categorical_tensor
)  # noqa: F821)
eager.register(Delta, Tensor, Tensor, Tensor)(eager_delta_tensor)  # noqa: F821
eager.register(Delta, Funsor, Funsor, Variable)(
    eager_delta_funsor_variable
)  # noqa: F821
eager.register(Delta, Variable, Funsor, Variable)(
    eager_delta_funsor_variable
)  # noqa: F821
eager.register(Delta, Variable, Funsor, Funsor)(eager_delta_funsor_funsor)  # noqa: F821
eager.register(Delta, Variable, Variable, Variable)(
    eager_delta_variable_variable
)  # noqa: F821
eager.register(Normal, Funsor, Tensor, Funsor)(eager_normal)  # noqa: F821
eager.register(MultivariateNormal, Funsor, Tensor, Funsor)(eager_mvn)  # noqa: F821
eager.register(
    Contraction, ops.LogaddexpOp, ops.AddOp, frozenset, Dirichlet, BernoulliProbs
)(  # noqa: F821
    eager_beta_bernoulli
)
eager.register(
    Contraction, ops.LogaddexpOp, ops.AddOp, frozenset, Dirichlet, Categorical
)(  # noqa: F821
    eager_dirichlet_categorical
)
eager.register(
    Contraction, ops.LogaddexpOp, ops.AddOp, frozenset, Dirichlet, Multinomial
)(  # noqa: F821
    eager_dirichlet_multinomial
)
eager.register(
    Contraction, ops.LogaddexpOp, ops.AddOp, frozenset, Gamma, Gamma
)(  # noqa: F821
    eager_gamma_gamma
)
eager.register(
    Contraction, ops.LogaddexpOp, ops.AddOp, frozenset, Gamma, Poisson
)(  # noqa: F821
    eager_gamma_poisson
)
eager.register(
    Binary, ops.SubOp, JointDirichletMultinomial, DirichletMultinomial
)(  # noqa: F821
    eager_dirichlet_posterior
)
eager.register(
    Reduce, ops.AddOp, Multinomial[Tensor, Funsor, Funsor], frozenset
)(  # noqa: F821
    eager_plate_multinomial
)
