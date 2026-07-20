
import inspect
import typing

from funsor import interpreter
from funsor.domains import BintType, Dependent, find_domain
from funsor.interpretations import eager
from funsor.interpreter import PatternMissingError
from funsor.tensor import Tensor
from funsor.terms import Binary, Number, Tuple, Unary, to_data, to_funsor
from funsor.typing import deep_issubclass
from funsor.util import as_callable

from . import ops


def eager_tensor_made_op(op, *args):
    pass


def make_op(fn):
    """
    Wrap a python function for use in Funsor.

    1. Creates a new ``Op`` subclass and instance ``op``.
    2. Registers a :func:`~funsor.domains.find_domain` rule based on ``fn``'s
       return type.
    3. Registers an eager rule.

    This assumes ``fn`` is compatible with broadcasting.
    """
    input_types = typing.get_type_hints(as_callable(fn))
    output_type = input_types.pop("return")
    parameters = tuple(inspect.Signature.from_callable(as_callable(fn)).parameters)
    hints = tuple(input_types.get(name) for name in parameters)
    arity = len(parameters)
    if arity == 1:
        op_cls = ops.UnaryOp
        funsor_cls = Unary
    elif arity == 2:
        op_cls = ops.BinaryOp
        funsor_cls = Binary
    else:
        raise NotImplementedError("TODO convert to a finitary")
    op = op_cls.make(fn)

    @find_domain.register(type(op))
    def find_domain_made_op(op, *args):
        pass

    pattern = [funsor_cls, type(op)] + [(Number, Tuple, Tensor)] * arity
    eager.register(*pattern)(eager_tensor_made_op)

    return op
