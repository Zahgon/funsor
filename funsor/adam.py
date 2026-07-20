
import funsor.ops as ops
from funsor.domains import RealsType
from funsor.interpretations import StatefulInterpretation
from funsor.tensor import Tensor
from funsor.terms import Funsor, Reduce
from funsor.util import get_backend


class Adam(StatefulInterpretation):

    def __init__(self, num_steps, **kwargs):
        name = kwargs.pop("name", "adam")
        super().__init__(name)
        self.num_steps = num_steps
        self.log_every = kwargs.pop("log_every", 0)
        self.optim_params = kwargs  # TODO make precise
        self.params = kwargs.pop("params", {})

    def param(self, name, domain=None):
        if name not in self.params:
            if domain is None:
                raise ValueError(f"Unknown param: {name}")
            self.params[name] = self._initialize(name, domain)
        return self.params[name]

    def _initialize(self, name, domain):
        assert isinstance(name, str)
        assert isinstance(domain, RealsType)
        if get_backend() == "torch":
            import torch

            return Tensor(torch.randn(domain.shape, requires_grad=True))
        raise NotImplementedError(f"Unsupported backend {get_backend()}")


@Adam.register(Reduce, ops.MinOp, Funsor, frozenset)
def adam_min(self, op, loss, reduced_vars):
    pass


@Adam.register(Reduce, ops.MaxOp, Funsor, frozenset)
def adam_max(self, op, loss, reduced_vars):
    pass
