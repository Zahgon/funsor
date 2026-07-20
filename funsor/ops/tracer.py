
from collections import OrderedDict
from functools import singledispatch

from .array import is_numeric_array
from .op import trace_ops
from .program import OpProgram


def _debug(x):
    pass


def trace_function(fn, kwargs: dict, *, allow_constants=False):
    """
    Traces function to an :class:`~funsor.ops.program.OpProgram` that runs on
    backend values.

    Example::

        # Create a function involving ops.
        def fn(a, b, x):
            return ops.add(ops.matmul(a, x), b)

        # Evaluate via Funsor substitution.
        data = dict(a=randn(3, 3), b=randn(3), x=randn(3))
        expected = fn(**data)

        # Alternatively evaluate via a program.
        program = trace_function(expr, data)
        actual = program(**data)
        assert (acutal == expected).all()

    :param Funsor expr: A funsor expression to evaluate.
    :returns: An op program.
    :rtype: ~funsor.ops.program.OpProgram
    """
    assert isinstance(kwargs, dict)
    assert all(is_variable(v) for v in kwargs.values())
    kwarg_ids = {id(v) for v in kwargs.values()}
    assert len(kwarg_ids) == len(kwargs), "repeated inputs"

    with trace_ops(is_variable) as trace:
        root = fn(**kwargs)
    assert is_variable(root)

    dag = OrderedDict({id(root): (root, None, None)})
    for result, op, args in reversed(trace.values()):  # backward
        if id(result) not in dag or not is_variable(result):
            continue  # not needed
        for arg in args:
            dag.setdefault(id(arg), (arg, None, None))
        dag[id(result)] = result, op, args
    anf = list(reversed(dag.values()))  # forward

    ids = {}
    constants = []
    for result, op, args in anf:
        if op is None and id(result) not in kwarg_ids:
            ids[id(result)] = len(ids)
            constants.append(result)
            if not allow_constants and is_variable(result):
                raise ValueError(f"Found constant: {repr(result)}")

    inputs = []
    for name, value in kwargs.items():
        ids[id(value)] = len(ids)
        inputs.append(name)

    operations = []
    for result, op, args in anf:
        if id(result) in ids:
            continue  # constant or free variable
        assert op is not None
        ids[id(result)] = len(ids)
        arg_ids = tuple(ids[id(arg)] for arg in args)
        operations.append((op, arg_ids))

    return OpProgram(constants, inputs, operations)


@singledispatch
def is_variable(x):
    """
    An object is variable if it is either backend arrays or is a nested tuple
    containing at least one backend array.
    """
    return is_numeric_array(x)


@is_variable.register(int)
def _is_variable_int(x):
    pass


@is_variable.register(float)
def _is_variable_float(x):
    pass


@is_variable.register(tuple)
def _is_variable_tuple(x):
    pass


__all__ = [
    "trace_function",
]
