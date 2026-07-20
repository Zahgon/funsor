
import functools
import itertools
from collections import Counter, OrderedDict, defaultdict
from functools import reduce
from typing import Tuple, Union

import opt_einsum

import funsor
import funsor.ops as ops
from funsor.affine import affine_inputs
from funsor.delta import Delta
from funsor.domains import find_domain
from funsor.gaussian import Gaussian
from funsor.interpretations import eager, normalize, reflect
from funsor.interpreter import children
from funsor.ops import DISTRIBUTIVE_OPS, AssociativeOp, NullOp
from funsor.tensor import Tensor
from funsor.terms import (
    _INFIX,
    Align,
    Binary,
    Funsor,
    Number,
    Reduce,
    Subs,
    Unary,
    Variable,
    to_funsor,
)
from funsor.typing import Variadic
from funsor.util import broadcast_shape, get_backend, quote


class Contraction(Funsor):

    def __init__(self, red_op, bin_op, reduced_vars, terms):
        terms = (terms,) if isinstance(terms, Funsor) else terms
        assert isinstance(red_op, AssociativeOp)
        assert isinstance(bin_op, AssociativeOp)
        assert all(isinstance(v, Funsor) for v in terms)
        assert isinstance(reduced_vars, frozenset)
        assert all(isinstance(v, Variable) for v in reduced_vars)
        assert isinstance(terms, tuple) and len(terms) > 0

        assert not (isinstance(red_op, NullOp) and isinstance(bin_op, NullOp))
        if isinstance(red_op, NullOp):
            assert not reduced_vars
        elif isinstance(bin_op, NullOp):
            assert len(terms) == 1
        else:
            assert reduced_vars and len(terms) > 1
            assert (red_op, bin_op) in DISTRIBUTIVE_OPS

        fresh = frozenset()
        bound = {v.name: v.output for v in reduced_vars}
        inputs = OrderedDict()
        for v in terms:
            inputs.update((k, d) for k, d in v.inputs.items() if k not in bound)

        if bin_op is ops.null:
            output = terms[0].output
        else:
            output = reduce(
                lambda lhs, rhs: find_domain(bin_op, lhs, rhs),
                [v.output for v in reversed(terms)],
            )
        super(Contraction, self).__init__(inputs, output, fresh, bound)
        self.red_op = red_op
        self.bin_op = bin_op
        self.terms = terms
        self.reduced_vars = reduced_vars

    def __repr__(self):
        if self.bin_op in _INFIX:
            bin_op = " " + _INFIX[self.bin_op] + " "
            return "{}.reduce({}, {})".format(
                bin_op.join(map(repr, self.terms)),
                self.red_op,
                str(set(self.reduced_vars)),
            )
        return super().__repr__()

    def __str__(self):
        if self.bin_op in _INFIX:
            bin_op = " " + _INFIX[self.bin_op] + " "
            return "({}).reduce({}, {})".format(
                bin_op.join(map(str, self.terms)),
                self.red_op,
                str(set(map(str, self.reduced_vars))),
            )
        return super().__str__()

    def _sample(self, sampled_vars, sample_inputs, rng_key):
        pass

    def align(self, names):
        assert isinstance(names, tuple)
        assert all(name in self.inputs for name in names)
        new_terms = tuple(
            t.align(tuple(n for n in names if n in t.inputs)) for t in self.terms
        )
        result = Contraction(self.red_op, self.bin_op, self.reduced_vars, *new_terms)
        if not names == tuple(result.inputs):
            return Align(
                result, names
            )  # raise NotImplementedError("TODO align all terms")
        return result

    def _alpha_convert(self, alpha_subs):
        reduced_vars = frozenset(
            to_funsor(alpha_subs.get(var.name, var), var.output)
            for var in self.reduced_vars
        )
        alpha_subs = {k: to_funsor(v, self.bound[k]) for k, v in alpha_subs.items()}
        red_op, bin_op, _, terms = super()._alpha_convert(alpha_subs)
        return red_op, bin_op, reduced_vars, terms


GaussianMixture = Contraction[
    Union[ops.LogaddexpOp, NullOp],
    ops.AddOp,
    frozenset,
    Tuple[Union[Tensor, Number], Gaussian],
]


@quote.register(Contraction)
def _(arg, indent, out):
    pass


@children.register(Contraction)
def children_contraction(x):
    pass


@children.register(Contraction)
def children_contraction(x):
    pass


@eager.register(Contraction, AssociativeOp, AssociativeOp, frozenset, Variadic[Funsor])
def eager_contraction_generic_to_tuple(red_op, bin_op, reduced_vars, *terms):
    pass


@eager.register(Contraction, AssociativeOp, AssociativeOp, frozenset, tuple)
def eager_contraction_generic_recursive(red_op, bin_op, reduced_vars, terms):
    pass


@eager.register(Contraction, AssociativeOp, AssociativeOp, frozenset, Funsor)
def eager_contraction_to_reduce(red_op, bin_op, reduced_vars, term):
    pass


@eager.register(Contraction, AssociativeOp, AssociativeOp, frozenset, Funsor, Funsor)
def eager_contraction_to_binary(red_op, bin_op, reduced_vars, lhs, rhs):
    pass


@eager.register(Contraction, ops.AddOp, ops.MulOp, frozenset, Tensor, Tensor)
def eager_contraction_tensor(red_op, bin_op, reduced_vars, *terms):
    pass


@eager.register(Contraction, ops.LogaddexpOp, ops.AddOp, frozenset, Tensor, Tensor)
def eager_contraction_tensor(red_op, bin_op, reduced_vars, *terms):
    pass


def _eager_contract_tensors(reduced_vars, terms, backend):
    pass


@eager.register(
    Contraction, ops.LogaddexpOp, ops.AddOp, frozenset, GaussianMixture, GaussianMixture
)
def eager_contraction_gaussian(red_op, bin_op, reduced_vars, x, y):
    pass


@affine_inputs.register(Contraction)
def _(fn):
    pass



ORDERING = {Delta: 1, Number: 2, Tensor: 3, Gaussian: 4, Unary[ops.NegOp, Gaussian]: 5}
GROUND_TERMS = tuple(ORDERING)


@normalize.register(
    Contraction, AssociativeOp, ops.AddOp, frozenset, GROUND_TERMS, GROUND_TERMS
)
def normalize_contraction_commutative_canonical_order(
    red_op, bin_op, reduced_vars, *terms
):
    pass


@normalize.register(
    Contraction, AssociativeOp, ops.AddOp, frozenset, GaussianMixture, GROUND_TERMS
)
def normalize_contraction_commute_joint(red_op, bin_op, reduced_vars, mixture, other):
    pass


@normalize.register(
    Contraction, AssociativeOp, ops.AddOp, frozenset, GROUND_TERMS, GaussianMixture
)
def normalize_contraction_commute_joint(red_op, bin_op, reduced_vars, other, mixture):
    pass


@normalize.register(
    Contraction, AssociativeOp, AssociativeOp, frozenset, Variadic[Funsor]
)
def normalize_contraction_generic_args(red_op, bin_op, reduced_vars, *terms):
    pass


@normalize.register(Contraction, NullOp, NullOp, frozenset, Funsor)
def normalize_trivial(red_op, bin_op, reduced_vars, term):
    pass


@normalize.register(Contraction, AssociativeOp, AssociativeOp, frozenset, tuple)
def normalize_contraction_generic_tuple(red_op, bin_op, reduced_vars, terms):
    pass




@normalize.register(Binary, AssociativeOp, Funsor, Funsor)
def binary_to_contract(op, lhs, rhs):
    pass


@normalize.register(Reduce, AssociativeOp, Funsor, frozenset)
def reduce_funsor(op, arg, reduced_vars):
    pass


@normalize.register(
    Unary,
    ops.NegOp,
    (Variable, Contraction[ops.AssociativeOp, ops.MulOp, frozenset, tuple]),
)
def unary_neg_variable(op, arg):
    pass




@normalize.register(Subs, Funsor, tuple)
def do_fresh_subs(arg, subs):
    pass


@normalize.register(Subs, Contraction, tuple)
def distribute_subs_contraction(arg, subs):
    pass


@normalize.register(Subs, Subs, tuple)
def normalize_fuse_subs(arg, subs):
    pass


@normalize.register(Binary, ops.SubOp, Funsor, Funsor)
def binary_subtract(op, lhs, rhs):
    pass


@normalize.register(Binary, ops.TruedivOp, Funsor, Funsor)
def binary_divide(op, lhs, rhs):
    pass


@normalize.register(Unary, ops.ExpOp, Unary[ops.LogOp, Funsor])
@normalize.register(Unary, ops.LogOp, Unary[ops.ExpOp, Funsor])
@normalize.register(Unary, ops.NegOp, Unary[ops.NegOp, Funsor])
@normalize.register(Unary, ops.ReciprocalOp, Unary[ops.ReciprocalOp, Funsor])
def unary_log_exp(op, arg):
    pass


@normalize.register(
    Unary, ops.ReciprocalOp, Contraction[NullOp, ops.MulOp, frozenset, tuple]
)
@normalize.register(Unary, ops.NegOp, Contraction[NullOp, ops.AddOp, frozenset, tuple])
def unary_contract(op, arg):
    pass


BACKEND_TO_EINSUM_BACKEND = {
    "numpy": "numpy",
    "torch": "torch",
    "jax": "jax.numpy",
}
BACKEND_TO_LOGSUMEXP_BACKEND = {
    "numpy": "funsor.einsum.numpy_log",
    "torch": "pyro.ops.einsum.torch_log",
    "jax": "funsor.einsum.numpy_log",
}
BACKEND_TO_MAP_BACKEND = {
    "numpy": "funsor.einsum.numpy_map",
    "torch": "pyro.ops.einsum.torch_map",
    "jax": "funsor.einsum.numpy_map",
}
