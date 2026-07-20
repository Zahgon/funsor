
import math
from collections import OrderedDict
from functools import reduce
from typing import Tuple, Union

from multipledispatch import dispatch

import funsor.ops as ops
from funsor.cnf import Contraction, GaussianMixture
from funsor.delta import Delta
from funsor.domains import Bint
from funsor.gaussian import Gaussian, align_gaussian
from funsor.interpretations import eager, moment_matching, normalize
from funsor.ops import AssociativeOp
from funsor.tensor import Tensor, align_tensor
from funsor.terms import Funsor, Independent, Number, Reduce, Unary
from funsor.typing import Variadic


@dispatch(str, str, Variadic[(Gaussian, GaussianMixture)])
def eager_cat_homogeneous(name, part_name, *parts):
    pass




@moment_matching.register(
    Contraction, AssociativeOp, AssociativeOp, frozenset, Variadic[object]
)
def moment_matching_contract_default(*args):
    pass


@moment_matching.register(
    Contraction, ops.LogaddexpOp, ops.AddOp, frozenset, (Number, Tensor), Gaussian
)
def moment_matching_contract_joint(red_op, bin_op, reduced_vars, discrete, gaussian):
    pass




@eager.register(Reduce, ops.AddOp, Unary[ops.ExpOp, Funsor], frozenset)
def eager_reduce_exp(op, arg, reduced_vars):
    pass


@eager.register(
    Independent,
    (
        Contraction[
            ops.NullOp,
            ops.AddOp,
            frozenset,
            Tuple[Delta, Union[Number, Tensor], Gaussian],
        ],
        Contraction[
            ops.NullOp,
            ops.AddOp,
            frozenset,
            Tuple[Delta, Union[Number, Tensor, Gaussian]],
        ],
    ),
    str,
    str,
    str,
)
def eager_independent_joint(joint, reals_var, bint_var, diag_var):
    pass
