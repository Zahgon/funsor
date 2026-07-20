

import functools
import warnings
import weakref
from collections import OrderedDict, namedtuple

import torch
from pyro.distributions import validation_enabled
from pyro.optim.clipped_adam import ClippedAdam as _ClippedAdam

import funsor


class Distribution(object):
    def __init__(self, funsor_dist, sample_inputs=None):
        assert isinstance(funsor_dist, funsor.Funsor)
        assert not sample_inputs or all(
            isinstance(inp.dtype, int) for inp in sample_inputs.values()
        )
        self.funsor_dist = funsor_dist
        self.output = self.funsor_dist.inputs["value"]
        self.sample_inputs = sample_inputs

    def log_prob(self, value):
        result = self.funsor_dist(value=value)
        if self.sample_inputs:
            result = result + funsor.tensor.Tensor(
                torch.zeros(*(size.dtype for size in self.sample_inputs.values())),
                self.sample_inputs,
            )
        return result

    def __call__(self):
        with funsor.interpretations.eager:
            dist = self.funsor_dist(value="value")
            delta = dist.sample(frozenset(["value"]), sample_inputs=self.sample_inputs)
        if isinstance(delta, funsor.cnf.Contraction):
            assert len(delta.terms) == 2
            assert any(isinstance(t, funsor.delta.Delta) for t in delta.terms)
            delta = [t for t in delta.terms if isinstance(t, funsor.delta.Delta)][0]
        assert isinstance(delta, funsor.delta.Delta)
        return delta.terms[0][1][0]

    def expand_inputs(self, name, size):
        if name in self.funsor_dist.inputs:
            assert self.funsor_dist.inputs[name] == funsor.Bint[int(size)]
            return self
        inputs = OrderedDict([(name, funsor.Bint[int(size)])])
        if self.sample_inputs:
            inputs.update(self.sample_inputs)
        return Distribution(self.funsor_dist, sample_inputs=inputs)



PYRO_STACK = []
PARAM_STORE = {}  # maps name -> (unconstrained_value, constraint)


def get_param_store():
    pass


class Messenger(object):
    def __init__(self, fn=None):
        self.fn = fn

    def __enter__(self):
        PYRO_STACK.append(self)

    def __exit__(self, *args, **kwargs):
        assert PYRO_STACK[-1] is self
        PYRO_STACK.pop()

    def process_message(self, msg):
        pass

    def postprocess_message(self, msg):
        pass

    def __call__(self, *args, **kwargs):
        with self:
            return self.fn(*args, **kwargs)


class trace(Messenger):
    def __enter__(self):
        super(trace, self).__enter__()
        self.trace = OrderedDict()
        return self.trace

    def postprocess_message(self, msg):
        assert (
            msg["type"] != "sample" or msg["name"] not in self.trace
        ), "sample sites must have unique names"
        self.trace[msg["name"]] = msg.copy()

    def get_trace(self, *args, **kwargs):
        pass


class replay(Messenger):
    def __init__(self, fn, guide_trace):
        self.guide_trace = guide_trace
        super(replay, self).__init__(fn)

    def process_message(self, msg):
        if msg["name"] in self.guide_trace:
            msg["value"] = self.guide_trace[msg["name"]]["value"]


class block(Messenger):
    def __init__(self, fn=None, hide_fn=lambda msg: True):
        self.hide_fn = hide_fn
        super(block, self).__init__(fn)

    def process_message(self, msg):
        if self.hide_fn(msg):
            msg["stop"] = True


class seed(Messenger):
    def __init__(self, fn=None, rng_seed=None):
        self.rng_seed = rng_seed
        super(seed, self).__init__(fn)

    def __enter__(self):
        self.old_rng_state = torch.get_rng_state()
        torch.manual_seed(self.rng_seed)

    def __exit__(self, type, value, traceback):
        torch.set_rng_state(self.old_rng_state)


CondIndepStackFrame = namedtuple("CondIndepStackFrame", ["name", "size", "dim"])


class PlateMessenger(Messenger):
    def __init__(self, fn, name, size, dim):
        assert dim < 0
        self.frame = CondIndepStackFrame(name, size, dim)
        super(PlateMessenger, self).__init__(fn)

    def process_message(self, msg):
        if msg["type"] in ("sample", "param"):
            assert self.frame.dim not in msg["cond_indep_stack"]
            msg["cond_indep_stack"][self.frame.dim] = self.frame
        if msg["type"] == "sample":
            msg["fn"] = msg["fn"].expand_inputs(self.frame.name, self.frame.size)


def tensor_to_funsor(value, cond_indep_stack, output):
    assert isinstance(value, torch.Tensor)
    event_shape = output.shape
    batch_shape = value.shape[: value.dim() - len(event_shape)]
    if torch._C._get_tracing_state():
        with funsor.tensor.ignore_jit_warnings():
            batch_shape = tuple(map(int, batch_shape))
    inputs = OrderedDict()
    data = value
    for dim, size in enumerate(batch_shape):
        if size == 1:
            data = data.squeeze(dim - value.dim())
        else:
            frame = cond_indep_stack[dim - len(batch_shape)]
            assert size == frame.size, (size, frame)
            inputs[frame.name] = funsor.Bint[int(size)]
    value = funsor.tensor.Tensor(data, inputs, output.dtype)
    assert value.output == output
    return value


class log_joint(Messenger):
    def __enter__(self):
        super(log_joint, self).__enter__()
        self.log_factors = OrderedDict()  # maps site name to log_prob factor
        self.plates = set()
        return self

    def process_message(self, msg):
        if msg["type"] == "sample":
            if msg["value"] is None:
                msg["value"] = funsor.Variable(msg["name"], msg["fn"].output)

    def postprocess_message(self, msg):
        if msg["type"] == "sample":
            assert (
                msg["name"] not in self.log_factors
            ), "all sites must have unique names"
            log_prob = msg["fn"].log_prob(msg["value"])
            self.log_factors[msg["name"]] = log_prob
            self.plates.update(f.name for f in msg["cond_indep_stack"].values())


def apply_stack(msg):
    for pointer, handler in enumerate(reversed(PYRO_STACK)):
        handler.process_message(msg)
        if msg.get("stop"):
            break
    if msg["value"] is None:
        msg["value"] = msg["fn"](*msg["args"])
    if isinstance(msg["value"], torch.Tensor):
        msg["value"] = tensor_to_funsor(
            msg["value"], msg["cond_indep_stack"], msg["output"]
        )

    for handler in PYRO_STACK[-pointer - 1 :]:
        handler.postprocess_message(msg)
    return msg


def sample(name, fn, obs=None, infer=None):
    fn = Distribution(fn)

    if not PYRO_STACK:
        return fn()

    initial_msg = {
        "type": "sample",
        "name": name,
        "fn": fn,
        "args": (),
        "value": obs,
        "cond_indep_stack": {},  # maps dim to CondIndepStackFrame
        "output": fn.output,
        "infer": {} if infer is None else infer,
    }

    msg = apply_stack(initial_msg)
    assert isinstance(msg["value"], funsor.Funsor)
    return msg["value"]


def param(
    name,
    init_value=None,
    constraint=torch.distributions.constraints.real,
    event_dim=None,
):
    cond_indep_stack = {}
    output = None
    if init_value is not None:
        if event_dim is None:
            event_dim = init_value.dim()
        output = funsor.Reals[init_value.shape[init_value.dim() - event_dim :]]

    def fn(init_value, constraint):
        if name in PARAM_STORE:
            unconstrained_value, constraint = PARAM_STORE[name]
        else:
            assert init_value is not None
            with torch.no_grad():
                constrained_value = init_value.detach()
                unconstrained_value = torch.distributions.transform_to(constraint).inv(
                    constrained_value
                )
            unconstrained_value.requires_grad_()
            unconstrained_value._funsor_metadata = (cond_indep_stack, output)
            PARAM_STORE[name] = unconstrained_value, constraint

        constrained_value = torch.distributions.transform_to(constraint)(
            unconstrained_value
        )
        constrained_value.unconstrained = weakref.ref(unconstrained_value)
        return tensor_to_funsor(
            constrained_value, *unconstrained_value._funsor_metadata
        )

    if not PYRO_STACK:
        return fn(init_value, constraint)

    initial_msg = {
        "type": "param",
        "name": name,
        "fn": fn,
        "args": (init_value, constraint),
        "value": None,
        "cond_indep_stack": cond_indep_stack,  # maps dim to CondIndepStackFrame
        "output": output,
    }

    msg = apply_stack(initial_msg)
    assert isinstance(msg["value"], funsor.Funsor)
    return msg["value"]


def plate(name, size, dim):
    return PlateMessenger(fn=None, name=name, size=size, dim=dim)


class PyroOptim(object):
    def __init__(self, optim_args):
        self.optim_args = optim_args
        self.optim_objs = {}

    def __call__(self, params):
        for param in params:
            if param in self.optim_objs:
                optim = self.optim_objs[param]
            else:
                optim = self.TorchOptimizer([param], **self.optim_args)
                self.optim_objs[param] = optim
            optim.step()


class Adam(PyroOptim):
    TorchOptimizer = torch.optim.Adam


class ClippedAdam(PyroOptim):
    TorchOptimizer = _ClippedAdam


class SVI(object):
    def __init__(self, model, guide, optim, loss):
        self.model = model
        self.guide = guide
        self.optim = optim
        self.loss = loss

    def step(self, *args, **kwargs):
        with trace() as param_capture:
            with block(hide_fn=lambda msg: msg["type"] != "param"):
                loss = self.loss(self.model, self.guide, *args, **kwargs)
        funsor.to_data(loss).backward()
        params = [site["value"].data.unconstrained() for site in param_capture.values()]
        self.optim(params)
        for p in params:
            p.grad = torch.zeros_like(p.grad)
        return loss.item()


def Expectation(log_probs, costs, sum_vars, prod_vars):
    result = 0
    for cost in costs:
        log_prob = funsor.sum_product.sum_product(
            sum_op=funsor.ops.logaddexp,
            prod_op=funsor.ops.add,
            factors=log_probs,
            plates=prod_vars,
            eliminate=(prod_vars | sum_vars) - frozenset(cost.inputs),
        )
        term = funsor.Integrate(log_prob, cost, sum_vars & frozenset(cost.inputs))
        term = term.reduce(funsor.ops.add, prod_vars & frozenset(cost.inputs))
        result += term
    return result


def elbo(model, guide, *args, **kwargs):
    with log_joint() as guide_log_joint:
        guide(*args, **kwargs)
    with log_joint() as model_log_joint:
        model(*args, **kwargs)

    guide_log_probs = list(guide_log_joint.log_factors.values())
    guide_aux_vars = (
        frozenset().union(*(f.inputs for f in guide_log_probs))
        - frozenset(guide_log_joint.plates)
        - frozenset(model_log_joint.log_factors)
    )
    if guide_aux_vars:
        guide_log_probs = funsor.sum_product.partial_sum_product(
            funsor.ops.logaddexp,
            funsor.ops.add,
            guide_log_probs,
            plates=frozenset(guide_log_joint.plates),
            eliminate=guide_aux_vars,
        )

    model_log_probs = list(model_log_joint.log_factors.values())
    model_aux_vars = (
        frozenset().union(*(f.inputs for f in model_log_probs))
        - frozenset(model_log_joint.plates)
        - frozenset(guide_log_joint.log_factors)
    )
    if model_aux_vars:
        model_log_probs = funsor.sum_product.partial_sum_product(
            funsor.ops.logaddexp,
            funsor.ops.add,
            model_log_probs,
            plates=frozenset(model_log_joint.plates),
            eliminate=model_aux_vars,
        )

    plates = frozenset().union(
        *(model_log_joint.plates.intersection(f.inputs) for f in model_log_probs)
    )
    plates = plates | frozenset().union(
        *(guide_log_joint.plates.intersection(f.inputs) for f in guide_log_probs)
    )
    sum_vars = frozenset().union(
        model_log_joint.log_factors, guide_log_joint.log_factors
    ) - frozenset(model_aux_vars | guide_aux_vars)

    costs = []
    log_probs = []
    for p in model_log_probs:
        costs.append(p)
    for q in guide_log_probs:
        costs.append(-q)
        log_probs.append(q)

    elbo = Expectation(
        tuple(log_probs), tuple(costs), sum_vars=sum_vars, prod_vars=plates
    )

    loss = -elbo
    assert not loss.inputs
    return loss


class ELBO(object):
    def __init__(self, **kwargs):
        self.options = kwargs

    def __call__(self, model, guide, *args, **kwargs):
        return elbo(model, guide, *args, **kwargs)


class Trace_ELBO(ELBO):
    def __call__(self, model, guide, *args, **kwargs):
        with funsor.montecarlo.MonteCarlo():
            return elbo(model, guide, *args, **kwargs)


class TraceMeanField_ELBO(ELBO):
    pass


class TraceEnum_ELBO(ELBO):
    def __call__(self, model, guide, *args, **kwargs):
        if self.options.get("optimize", None):
            with funsor.optimizer.optimize:
                elbo_expr = elbo(model, guide, *args, **kwargs)
            return funsor.reinterpret(elbo_expr)
        return elbo(model, guide, *args, **kwargs)


class Jit(object):
    def __init__(self, fn, **kwargs):
        self.fn = fn
        self.ignore_jit_warnings = kwargs.get("ignore_jit_warnings", False)
        self._compiled = None
        self._param_trace = None

    def __call__(self, *args):
        if self._param_trace is None:
            with block(), trace() as tr, block(hide_fn=lambda m: m["type"] != "param"):
                self.fn(*args)
            self._param_trace = tr

        unconstrained_params = tuple(
            param(name).data.unconstrained() for name in self._param_trace
        )
        params_and_args = unconstrained_params + args

        if self._compiled is None:

            def compiled(*params_and_args):
                pass

            with validation_enabled(False), warnings.catch_warnings():
                if self.ignore_jit_warnings:
                    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                self._compiled = torch.jit.trace(
                    compiled, params_and_args, check_trace=False
                )

        data = self._compiled(*params_and_args)
        return funsor.tensor.Tensor(data)


class Jit_ELBO(ELBO):
    def __init__(self, elbo, **kwargs):
        super(Jit_ELBO, self).__init__(**kwargs)
        self._elbo = elbo(**kwargs)
        self._compiled = {}  # maps (model,guide) -> Jit instances

    def __call__(self, model, guide, *args):
        if (model, guide) not in self._compiled:
            elbo = functools.partial(self._elbo, model, guide)
            self._compiled[model, guide] = Jit(elbo, **self.options)
        return self._compiled[model, guide](*args)


def JitTrace_ELBO(**kwargs):
    pass


def JitTraceMeanField_ELBO(**kwargs):
    pass


def JitTraceEnum_ELBO(**kwargs):
    pass
