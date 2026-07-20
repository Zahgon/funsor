
from collections import defaultdict
from collections.abc import Hashable

from funsor.cnf import Contraction
from funsor.interpretations import Interpretation, reflect
from funsor.interpreter import stack_reinterpret
from funsor.ops import AssociativeOp
from funsor.registry import KeyedRegistry
from funsor.terms import (
    Approximate,
    Binary,
    Cat,
    Funsor,
    Reduce,
    Scatter,
    Slice,
    Subs,
    substitute,
    to_funsor,
)

from . import instrument, interpreter, ops


def _alpha_unmangle(expr):
    alpha_subs = {
        name: name.split("__BOUND")[0] for name in expr.bound if "__BOUND" in name
    }
    if not alpha_subs:
        return tuple(expr._ast_values)

    return expr._alpha_convert(alpha_subs)


class AdjointTape(Interpretation):
    def __init__(self):
        super().__init__("adjoint")
        self.tape = []
        self._old_interpretation = None
        self._eager_to_lazy = {}

    def interpret(self, cls, *args):
        if cls in adjoint_ops:  # atomic op, don't trace internals
            with self._old_interpretation:
                result = cls(*args)
            self.tape.append((result, cls, args))
        else:
            result = self._old_interpretation.interpret(cls, *args)
        lazy_args = [
            self._eager_to_lazy.get(
                (
                    id(arg)
                    if ops.is_numeric_array(arg) or not isinstance(arg, Hashable)
                    else arg
                ),
                arg,
            )
            for arg in args
        ]
        with self._old_interpretation:
            self._eager_to_lazy[result] = reflect.interpret(cls, *lazy_args)
        return result

    def __enter__(self):
        self.tape = []
        self._old_interpretation = interpreter.get_interpretation()
        return super().__enter__()

    def adjoint(self, sum_op, bin_op, root, targets=None, *, batch_vars=set()):
        zero = to_funsor(ops.UNITS[sum_op])
        one = to_funsor(ops.UNITS[bin_op])
        adjoint_values = defaultdict(lambda: zero)
        adjoint_values[root] = one

        reached_root = False
        while self.tape:
            output, fn, inputs = self.tape.pop()
            if not reached_root:
                if output is root:
                    reached_root = True
                else:
                    continue

            with reflect:
                lazy_output = self._eager_to_lazy[output]
                lazy_fn = type(lazy_output)
                lazy_inputs = lazy_output._ast_values
                lazy_other_subs = tuple(
                    (name, to_funsor(name.split("__BOUND")[0], domain))
                    for name, domain in lazy_output.inputs.items()
                    if "__BOUND" in name
                )
                lazy_inputs = _alpha_unmangle(
                    substitute(lazy_fn(*lazy_inputs), lazy_other_subs)
                )
                lazy_output = type(lazy_output)(
                    *_alpha_unmangle(substitute(lazy_output, lazy_other_subs))
                )

                other_subs = tuple(
                    (name, to_funsor(name.split("__BOUND")[0], domain))
                    for name, domain in output.inputs.items()
                    if "__BOUND" in name
                )
                inputs = _alpha_unmangle(substitute(fn(*inputs), other_subs))
                output = type(output)(*_alpha_unmangle(substitute(output, other_subs)))

                self._eager_to_lazy[output] = lazy_output

            in_adjs = adjoint_ops(fn, sum_op, bin_op, adjoint_values[output], *inputs)
            for v, adjv in in_adjs:
                agg_vars = adjv.input_vars - v.input_vars - root.input_vars - batch_vars
                assert "particle" not in {var.name for var in agg_vars}  # DEBUG FIXME
                old_value = adjoint_values[v]
                adjoint_values[v] = sum_op(old_value, adjv.reduce(sum_op, agg_vars))

        result = defaultdict(lambda: zero)
        for key, value in adjoint_values.items():
            lazy_key = self._eager_to_lazy.get(key, key)
            result[lazy_key] = value

        if targets is None:
            return result
        return {target: result[target] for target in targets}


def forward_backward(sum_op, bin_op, expr, *, batch_vars=frozenset()):
    with AdjointTape() as tape:
        forward = stack_reinterpret(expr)
    backward = tape.adjoint(sum_op, bin_op, forward, batch_vars=batch_vars)
    return forward, backward


def adjoint(sum_op, bin_op, expr):
    forward, backward = forward_backward(sum_op, bin_op, expr)
    return backward


def _fail_default(*args):
    raise NotImplementedError("Should not be here! {}".format(args))


adjoint_ops = KeyedRegistry(default=_fail_default)
if instrument.DEBUG:
    adjoint_ops_register = adjoint_ops.register
    adjoint_ops.register = lambda *args: lambda fn: adjoint_ops_register(*args)(
        instrument.debug_logged(fn)
    )


@adjoint_ops.register(
    Binary, AssociativeOp, AssociativeOp, Funsor, AssociativeOp, Funsor, Funsor
)
def adjoint_binary(adj_sum_op, adj_prod_op, out_adj, op, lhs, rhs):
    pass


@adjoint_ops.register(
    Reduce, AssociativeOp, AssociativeOp, Funsor, AssociativeOp, Funsor, frozenset
)
def adjoint_reduce(adj_sum_op, adj_prod_op, out_adj, op, arg, reduced_vars):
    pass


@adjoint_ops.register(
    Contraction,
    AssociativeOp,
    AssociativeOp,
    Funsor,
    AssociativeOp,
    AssociativeOp,
    frozenset,
    Funsor,
)
def adjoint_contract_unary(
    adj_sum_op, adj_prod_op, out_adj, sum_op, prod_op, reduced_vars, arg
):
    pass


@adjoint_ops.register(
    Contraction,
    AssociativeOp,
    AssociativeOp,
    Funsor,
    AssociativeOp,
    AssociativeOp,
    frozenset,
    tuple,
)
def adjoint_contract_generic(
    adj_sum_op, adj_prod_op, out_adj, sum_op, prod_op, reduced_vars, terms
):
    pass


@adjoint_ops.register(
    Contraction,
    AssociativeOp,
    AssociativeOp,
    Funsor,
    AssociativeOp,
    AssociativeOp,
    frozenset,
    Funsor,
    Funsor,
)
def adjoint_contract(
    adj_sum_op, adj_prod_op, out_adj, sum_op, prod_op, reduced_vars, lhs, rhs
):
    pass


@adjoint_ops.register(Cat, AssociativeOp, AssociativeOp, Funsor, str, tuple, str)
def adjoint_cat(adj_sum_op, adj_prod_op, out_adj, name, parts, part_name):
    pass


@adjoint_ops.register(Subs, AssociativeOp, AssociativeOp, Funsor, Funsor, tuple)
def adjoint_subs(adj_sum_op, adj_prod_op, out_adj, arg, subs):
    pass


@adjoint_ops.register(
    Scatter,
    AssociativeOp,
    AssociativeOp,
    Funsor,
    AssociativeOp,
    tuple,
    Funsor,
    frozenset,
)
def adjoint_scatter(adj_sum_op, adj_prod_op, out_adj, op, subs, source, reduced_vars):
    pass
