
from collections import OrderedDict

import torch

import funsor.ops as ops
from funsor.domains import Bint, Reals
from funsor.interpretations import eager, lazy, moment_matching
from funsor.pyro.convert import (
    dist_to_funsor,
    funsor_to_cat_and_mvn,
    funsor_to_tensor,
    matrix_and_mvn_to_funsor,
    mvn_to_funsor,
    tensor_to_funsor,
)
from funsor.pyro.distribution import FunsorDistribution
from funsor.sum_product import (
    MarkovProduct,
    naive_sequential_sum_product,
    sequential_sum_product,
)
from funsor.terms import Variable
from funsor.util import broadcast_shape


class DiscreteHMM(FunsorDistribution):

    def __init__(
        self, initial_logits, transition_logits, observation_dist, validate_args=None
    ):
        assert isinstance(initial_logits, torch.Tensor)
        assert isinstance(transition_logits, torch.Tensor)
        assert isinstance(observation_dist, torch.distributions.Distribution)
        assert initial_logits.dim() >= 1
        assert transition_logits.dim() >= 2
        assert len(observation_dist.batch_shape) >= 1
        shape = broadcast_shape(
            initial_logits.shape[:-1] + (1,),
            transition_logits.shape[:-2],
            observation_dist.batch_shape[:-1],
        )
        batch_shape, time_shape = shape[:-1], shape[-1:]
        event_shape = time_shape + observation_dist.event_shape
        self._has_rsample = observation_dist.has_rsample

        initial_logits = initial_logits - initial_logits.logsumexp(-1, True)
        transition_logits = transition_logits - transition_logits.logsumexp(-1, True)

        init = tensor_to_funsor(initial_logits, ("state",))
        trans = tensor_to_funsor(transition_logits, ("time", "state", "state(time=1)"))
        obs = dist_to_funsor(observation_dist, ("time", "state(time=1)"))
        dtype = obs.inputs["value"].dtype

        with lazy:
            funsor_dist = Variable("value", obs.inputs["value"])  # a bogus value
            self._init = init
            self._trans = trans
            self._obs = obs

        super(DiscreteHMM, self).__init__(
            funsor_dist, batch_shape, event_shape, dtype, validate_args
        )

    @torch.distributions.constraints.dependent_property
    def has_rsample(self):
        pass

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        ndims = max(len(self.batch_shape), value.dim() - self.event_dim)
        time = Variable("time", Bint[self.event_shape[0]])
        value = tensor_to_funsor(
            value, ("time",), event_output=self.event_dim - 1, dtype=self.dtype
        )

        obs = self._obs(value=value)
        result = self._trans + obs
        result = sequential_sum_product(
            ops.logaddexp, ops.add, result, time, {"state": "state(time=1)"}
        )
        result = self._init + result.reduce(ops.logaddexp, "state(time=1)")
        result = result.reduce(ops.logaddexp, "state")

        result = funsor_to_tensor(result, ndims=ndims)
        return result

    def _sample_delta(self, sample_shape):
        raise NotImplementedError("TODO")

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(DiscreteHMM, _instance)
        batch_shape = torch.Size(batch_shape)
        new._has_rsample = self._has_rsample
        new._init = self._init + tensor_to_funsor(torch.zeros(batch_shape))
        new._trans = self._trans
        new._obs = self._obs
        super(DiscreteHMM, new).__init__(
            self.funsor_dist,
            batch_shape,
            self.event_shape,
            self.dtype,
            validate_args=False,
        )
        new.validate_args = self.__dict__.get("_validate_args")
        return new


class GaussianHMM(FunsorDistribution):

    has_rsample = True
    arg_constraints = {}

    def __init__(
        self,
        initial_dist,
        transition_matrix,
        transition_dist,
        observation_matrix,
        observation_dist,
        validate_args=None,
    ):
        assert isinstance(initial_dist, torch.distributions.MultivariateNormal)
        assert isinstance(transition_matrix, torch.Tensor)
        assert isinstance(transition_dist, torch.distributions.MultivariateNormal)
        assert isinstance(observation_matrix, torch.Tensor)
        assert isinstance(observation_dist, torch.distributions.MultivariateNormal)
        hidden_dim, obs_dim = observation_matrix.shape[-2:]
        assert obs_dim >= hidden_dim // 2, "obs_dim must be at least half of hidden_dim"
        assert initial_dist.event_shape == (hidden_dim,)
        assert transition_matrix.shape[-2:] == (hidden_dim, hidden_dim)
        assert transition_dist.event_shape == (hidden_dim,)
        assert observation_dist.event_shape == (obs_dim,)
        shape = broadcast_shape(
            initial_dist.batch_shape + (1,),
            transition_matrix.shape[:-2],
            transition_dist.batch_shape,
            observation_matrix.shape[:-2],
            observation_dist.batch_shape,
        )
        batch_shape, time_shape = shape[:-1], shape[-1:]
        event_shape = time_shape + (obs_dim,)

        init = dist_to_funsor(initial_dist)(value="state")
        trans = matrix_and_mvn_to_funsor(
            transition_matrix, transition_dist, ("time",), "state", "state(time=1)"
        )
        obs = matrix_and_mvn_to_funsor(
            observation_matrix, observation_dist, ("time",), "state(time=1)", "value"
        )
        dtype = "real"

        with lazy:
            value = Variable("value", Reals[time_shape[0], obs_dim])
            result = trans + obs(value=value["time"])
            result = MarkovProduct(
                ops.logaddexp, ops.add, result, "time", {"state": "state(time=1)"}
            )
            result = init + result.reduce(ops.logaddexp, "state(time=1)")
            funsor_dist = result.reduce(ops.logaddexp, "state")

        super(GaussianHMM, self).__init__(
            funsor_dist, batch_shape, event_shape, dtype, validate_args
        )
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim


class GaussianMRF(FunsorDistribution):

    has_rsample = True

    def __init__(
        self, initial_dist, transition_dist, observation_dist, validate_args=None
    ):
        assert isinstance(initial_dist, torch.distributions.MultivariateNormal)
        assert isinstance(transition_dist, torch.distributions.MultivariateNormal)
        assert isinstance(observation_dist, torch.distributions.MultivariateNormal)
        hidden_dim = initial_dist.event_shape[0]
        assert transition_dist.event_shape[0] == hidden_dim + hidden_dim
        obs_dim = observation_dist.event_shape[0] - hidden_dim
        shape = broadcast_shape(
            initial_dist.batch_shape + (1,),
            transition_dist.batch_shape,
            observation_dist.batch_shape,
        )
        batch_shape, time_shape = shape[:-1], shape[-1:]
        event_shape = time_shape + (obs_dim,)

        init = dist_to_funsor(initial_dist)(value="state")
        trans = mvn_to_funsor(
            transition_dist,
            ("time",),
            OrderedDict(
                [("state", Reals[hidden_dim]), ("state(time=1)", Reals[hidden_dim])]
            ),
        )
        obs = mvn_to_funsor(
            observation_dist,
            ("time",),
            OrderedDict(
                [("state(time=1)", Reals[hidden_dim]), ("value", Reals[obs_dim])]
            ),
        )

        with lazy:
            time = Variable("time", Bint[time_shape[0]])
            value = Variable("value", Reals[time_shape[0], obs_dim])
            logp_oh = trans + obs(value=value["time"])
            logp_oh = MarkovProduct(
                ops.logaddexp, ops.add, logp_oh, time, {"state": "state(time=1)"}
            )
            logp_oh += init
            logp_oh = logp_oh.reduce(
                ops.logaddexp, frozenset({"state", "state(time=1)"})
            )
            logp_h = trans + obs.reduce(ops.logaddexp, "value")
            logp_h = MarkovProduct(
                ops.logaddexp, ops.add, logp_h, time, {"state": "state(time=1)"}
            )
            logp_h += init
            logp_h = logp_h.reduce(ops.logaddexp, frozenset({"state", "state(time=1)"}))
            funsor_dist = logp_oh - logp_h

        dtype = "real"
        super(GaussianMRF, self).__init__(
            funsor_dist, batch_shape, event_shape, dtype, validate_args
        )
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim


class SwitchingLinearHMM(FunsorDistribution):

    has_rsample = True
    arg_constraints = {}

    def __init__(
        self,
        initial_logits,
        initial_mvn,
        transition_logits,
        transition_matrix,
        transition_mvn,
        observation_matrix,
        observation_mvn,
        exact=False,
        validate_args=None,
    ):
        assert isinstance(initial_logits, torch.Tensor)
        assert isinstance(initial_mvn, torch.distributions.MultivariateNormal)
        assert isinstance(transition_logits, torch.Tensor)
        assert isinstance(transition_matrix, torch.Tensor)
        assert isinstance(transition_mvn, torch.distributions.MultivariateNormal)
        assert isinstance(observation_matrix, torch.Tensor)
        assert isinstance(observation_mvn, torch.distributions.MultivariateNormal)
        hidden_cardinality = initial_logits.size(-1)
        hidden_dim, obs_dim = observation_matrix.shape[-2:]
        assert obs_dim >= hidden_dim // 2, "obs_dim must be at least half of hidden_dim"
        assert initial_mvn.event_shape[0] == hidden_dim
        assert transition_logits.size(-1) == hidden_cardinality
        assert transition_matrix.shape[-2:] == (hidden_dim, hidden_dim)
        assert transition_mvn.event_shape[0] == hidden_dim
        assert observation_mvn.event_shape[0] == obs_dim
        init_shape = broadcast_shape(initial_logits.shape, initial_mvn.batch_shape)
        shape = broadcast_shape(
            init_shape[:-1] + (1, init_shape[-1]),
            transition_logits.shape[:-1],
            transition_matrix.shape[:-2],
            transition_mvn.batch_shape,
            observation_matrix.shape[:-2],
            observation_mvn.batch_shape,
        )
        assert shape[-1] == hidden_cardinality
        batch_shape, time_shape = shape[:-2], shape[-2:-1]
        event_shape = time_shape + (obs_dim,)

        initial_logits = initial_logits - initial_logits.logsumexp(-1, True)
        transition_logits = transition_logits - transition_logits.logsumexp(-1, True)

        init = tensor_to_funsor(initial_logits, ("class",)) + dist_to_funsor(
            initial_mvn, ("class",)
        )(value="state")
        trans = tensor_to_funsor(
            transition_logits, ("time", "class", "class(time=1)")
        ) + matrix_and_mvn_to_funsor(
            transition_matrix,
            transition_mvn,
            ("time", "class(time=1)"),
            "state",
            "state(time=1)",
        )
        obs = matrix_and_mvn_to_funsor(
            observation_matrix,
            observation_mvn,
            ("time", "class(time=1)"),
            "state(time=1)",
            "value",
        )
        if "class(time=1)" not in set(trans.inputs).union(obs.inputs):
            raise ValueError(
                "neither transition nor observation depend on discrete state"
            )
        dtype = "real"

        with lazy:
            funsor_dist = Variable("value", obs.inputs["value"])  # a bogus value
            self._init = init
            self._trans = trans
            self._obs = obs

        super(SwitchingLinearHMM, self).__init__(
            funsor_dist, batch_shape, event_shape, dtype, validate_args
        )
        self.exact = exact

    def log_prob(self, value):
        ndims = max(len(self.batch_shape), value.dim() - 2)
        time = Variable("time", Bint[self.event_shape[0]])
        value = tensor_to_funsor(value, ("time",), 1)

        seq_sum_prod = (
            naive_sequential_sum_product if self.exact else sequential_sum_product
        )
        with eager if self.exact else moment_matching:
            result = self._trans + self._obs(value=value)
            result = seq_sum_prod(
                ops.logaddexp,
                ops.add,
                result,
                time,
                {"class": "class(time=1)", "state": "state(time=1)"},
            )
            result += self._init
            result = result.reduce(
                ops.logaddexp,
                frozenset(["class", "state", "class(time=1)", "state(time=1)"]),
            )

            result = funsor_to_tensor(result, ndims=ndims)
            return result

    def _sample_delta(self, sample_shape):
        raise NotImplementedError("TODO")

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(SwitchingLinearHMM, _instance)
        batch_shape = torch.Size(batch_shape)
        new._init = self._init + tensor_to_funsor(torch.zeros(batch_shape))
        new._trans = self._trans
        new._obs = self._obs
        new.exact = self.exact
        super(SwitchingLinearHMM, new).__init__(
            self.funsor_dist,
            batch_shape,
            self.event_shape,
            self.dtype,
            validate_args=False,
        )
        new.validate_args = self.__dict__.get("_validate_args")
        return new

    def filter(self, value):
        pass
