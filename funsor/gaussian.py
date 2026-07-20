
import math
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from functools import reduce

import funsor.ops as ops
from funsor.affine import affine_inputs, extract_affine, is_affine
from funsor.delta import Delta
from funsor.domains import Real, Reals
from funsor.interpretations import compress_gaussians
from funsor.ops import AddOp, SubOp
from funsor.tensor import Tensor, align_tensor, align_tensors
from funsor.terms import (
    Binary,
    Funsor,
    FunsorMeta,
    Number,
    Slice,
    Subs,
    Variable,
    eager,
    reflect,
)
from funsor.util import broadcast_shape, get_tracing_state, lazy_property


def _log_det_tri(x):
    pass


def _vv(vec1, vec2):
    """
    Computes the inner product ``< vec1 | vec 2 >``.
    """
    return (vec1[..., None, :] @ vec2[..., None])[..., 0, 0]


def _norm2(vec):
    return _vv(vec, vec)


def _mv(mat, vec):
    pass


def _vm(vec, mat):
    return (vec[..., None, :] @ mat)[..., 0, :]


def _mmt(mat1, mat2=None):
    if mat2 is None:
        mat2 = mat1
    return mat1 @ ops.transpose(mat2, -1, -2)


def _mtm(mat1, mat2=None):
    pass


def _inverse_cholesky(P):
    """
    Computes a Cholesky decomposition of the inverse of a posdef matrix.
    """
    Lf = ops.cholesky(ops.flip(P, (-2, -1)))
    L_inv = ops.transpose(ops.flip(Lf, (-2, -1)), -2, -1)
    L = ops.triangular_inv(L_inv)
    return L


def _compress_rank(white_vec, prec_sqrt, assume_full_rank=False):
    """
    Compress a wide representation ``(white_vec, prec_sqrt)`` while preserving
    the quadratic function ``||x @ prec_sqrt - white_vec||^2 + const``.
    """
    dim, rank = prec_sqrt.shape[-2:]
    assert rank >= dim
    old_norm2 = _norm2(white_vec)

    if assume_full_rank:
        info_vec_ = prec_sqrt @ white_vec[..., None]
        precision = prec_sqrt @ ops.transpose(prec_sqrt, -1, -2)
        prec_sqrt = ops.cholesky(precision)
        white_vec = ops.triangular_solve(info_vec_, prec_sqrt)[..., 0]
    else:
        Q, R = ops.qr(ops.transpose(prec_sqrt, -1, -2))
        assert Q.shape[-2:] == (rank, dim)  # note only part of Q is returned
        assert R.shape[-2:] == (dim, dim)
        prec_sqrt = ops.transpose(R, -1, -2)
        white_vec = _vm(white_vec, Q)
    new_norm2 = _norm2(white_vec)
    shift = 0.5 * (new_norm2 - old_norm2)
    return white_vec, prec_sqrt, shift


def _compute_offsets(inputs):
    """
    Compute offsets of real inputs into the concatenated Gaussian dims.
    This ignores all int inputs.

    :param OrderedDict inputs: A schema mapping variable name to domain.
    :return: a pair ``(offsets, total)``, where ``offsets`` is an OrderedDict
        mapping input name to integer offset, and ``total`` is the total event
        size.
    :rtype: tuple
    """
    assert isinstance(inputs, OrderedDict)
    offsets = OrderedDict()
    total = 0
    for key, domain in inputs.items():
        if domain.dtype == "real":
            offsets[key] = total
            total += domain.num_elements
    return offsets, total


def _split_real_inputs(inputs, lhs_keys, prototype):
    """
    Finds a splitting set of indices ``(lhs, rhs)`` into the flat real
    dimension such that ``lhs`` indexes into real inputs in ``lhs_keys`` and
    ``rhs`` indexes into everything else.
    """
    lhs_blocks = []
    rhs_blocks = []
    start = 0
    for key, domain in inputs.items():
        if domain.dtype == "real":
            stop = start + domain.num_elements
            (lhs_blocks if key in lhs_keys else rhs_blocks).append(slice(start, stop))
            start = stop

    lhs_start = min(b.start for b in lhs_blocks)
    rhs_start = min(b.start for b in rhs_blocks)
    lhs_stop = max(b.stop for b in lhs_blocks)
    rhs_stop = max(b.stop for b in rhs_blocks)
    if lhs_stop <= rhs_start or rhs_stop <= lhs_start:
        lhs = slice(lhs_start, lhs_stop)
        rhs = slice(rhs_start, rhs_stop)
        return lhs, rhs

    lhs = ops.cat([ops.new_arange(prototype, b.start, b.stop) for b in lhs_blocks])
    rhs = ops.cat([ops.new_arange(prototype, b.start, b.stop) for b in rhs_blocks])
    return lhs, rhs


def _find_intervals(intervals, end):
    """
    Finds a complete set of intervals partitioning [0, end), given a partial
    set of non-overlapping intervals.
    """
    cuts = list(sorted({0, end}.union(*intervals)))
    return list(zip(cuts[:-1], cuts[1:]))


def _parse_slices(index, value):
    pass


class BlockVector(object):

    def __init__(self, shape):
        self.shape = shape
        self.parts = {}

    def __setitem__(self, index, value):
        (i,), value = _parse_slices(index, value)
        self.parts[i] = value

    def as_tensor(self):

        prototype = next(iter(self.parts.values()))
        for i in _find_intervals(self.parts.keys(), self.shape[-1]):
            if i not in self.parts:
                self.parts[i] = ops.new_zeros(
                    prototype, self.shape[:-1] + (i[1] - i[0],)
                )

        parts = [v for k, v in sorted(self.parts.items())]
        result = ops.cat(parts, -1)
        if not get_tracing_state():
            assert result.shape == self.shape
        return result


class BlockMatrix(object):

    def __init__(self, shape):
        self.shape = shape
        self.parts = defaultdict(dict)

    def __setitem__(self, index, value):
        (i, j), value = _parse_slices(index, value)
        self.parts[i][j] = value

    def as_tensor(self):

        arbitrary_row = next(iter(self.parts.values()))
        prototype = next(iter(arbitrary_row.values()))
        js = set().union(*(part.keys() for part in self.parts.values()))
        rows = _find_intervals(self.parts.keys(), self.shape[-2])
        cols = _find_intervals(js, self.shape[-1])
        for i in rows:
            for j in cols:
                if j not in self.parts[i]:
                    shape = self.shape[:-2] + (i[1] - i[0], j[1] - j[0])
                    self.parts[i][j] = ops.new_zeros(prototype, shape)

        columns = {
            i: ops.cat([v for j, v in sorted(part.items())], -1)
            for i, part in self.parts.items()
        }
        result = ops.cat([v for i, v in sorted(columns.items())], -2)
        if not get_tracing_state():
            assert result.shape == self.shape
        return result


def align_gaussian(new_inputs, old, expand=False):
    """
    Align data of a Gaussian distribution to a new ``inputs`` shape.
    """
    assert isinstance(new_inputs, OrderedDict)
    assert isinstance(old, Gaussian)
    white_vec = old.white_vec
    prec_sqrt = old.prec_sqrt

    new_ints = OrderedDict((k, d) for k, d in new_inputs.items() if d.dtype != "real")
    old_ints = OrderedDict((k, d) for k, d in old.inputs.items() if d.dtype != "real")
    if new_ints != old_ints:
        white_vec = align_tensor(new_ints, Tensor(white_vec, old_ints), expand=expand)
        prec_sqrt = align_tensor(new_ints, Tensor(prec_sqrt, old_ints), expand=expand)

    new_offsets, new_dim = _compute_offsets(new_inputs)
    old_offsets, old_dim = _compute_offsets(old.inputs)
    assert prec_sqrt.shape[-2:-1] == (old_dim,)
    if new_offsets != old_offsets:
        old_prec_sqrt = ops.transpose(prec_sqrt, -1, -2)
        prec_sqrt = BlockVector(old_prec_sqrt.shape[:-1] + (new_dim,))
        for k, new_offset in new_offsets.items():
            if k not in old_offsets:
                continue
            offset = old_offsets[k]
            num_elements = old.inputs[k].num_elements
            old_slice = slice(offset, offset + num_elements)
            new_slice = slice(new_offset, new_offset + num_elements)
            prec_sqrt[..., new_slice] = old_prec_sqrt[..., old_slice]
        prec_sqrt = prec_sqrt.as_tensor()
        prec_sqrt = ops.transpose(prec_sqrt, -1, -2)

    return white_vec, prec_sqrt


class GaussianMeta(FunsorMeta):

    def __call__(
        cls,
        white_vec=None,
        prec_sqrt=None,
        inputs=None,
        *,
        mean=None,
        info_vec=None,
        precision=None,
        scale_tril=None,
        covariance=None,
    ):
        assert inputs is not None
        if isinstance(inputs, OrderedDict):
            inputs = tuple(inputs.items())
        assert isinstance(inputs, tuple)

        if prec_sqrt is None and white_vec is not None:
            raise ValueError("Cannot specify white_vec without prec_sqrt")
        if prec_sqrt is not None:
            is_tril = False
        elif precision is not None:
            prec_sqrt = ops.cholesky(precision)
            is_tril = True
        elif covariance is not None:
            prec_sqrt = _inverse_cholesky(covariance)
            is_tril = True
        elif scale_tril is not None:
            prec_sqrt = ops.transpose(ops.triangular_inv(scale_tril), -1, -2)
            is_tril = False
        else:
            raise ValueError(
                "At least one of prec_sqrt, precision, scale_tril, or covariance "
                "must be specified"
            )

        if white_vec is not None:
            pass
        elif mean is not None:
            white_vec = _vm(mean, prec_sqrt)
        elif info_vec is not None:
            if not is_tril:
                prec_sqrt = ops.cholesky(_mmt(prec_sqrt))  # triangularize
                is_tril = True
            white_vec = ops.triangular_solve(info_vec[..., None], prec_sqrt)[..., 0]
        else:
            raise ValueError(
                "At least one of white_vec, mean, or info_vec must be specified"
            )

        shift = None
        dim, rank = prec_sqrt.shape[-2:]
        if rank > dim * cls.compression_threshold:
            white_vec, prec_sqrt, shift = _compress_rank(white_vec, prec_sqrt)

        result = super().__call__(white_vec, prec_sqrt, inputs)

        if shift is not None:
            int_inputs = OrderedDict((k, v) for k, v in inputs if v.dtype != "real")
            result += Tensor(shift, int_inputs)

        return result


class Gaussian(Funsor, metaclass=GaussianMeta):

    compression_threshold = 2

    def __init__(self, white_vec, prec_sqrt, inputs):
        assert ops.is_numeric_array(white_vec) and ops.is_numeric_array(prec_sqrt)
        assert isinstance(inputs, tuple)
        inputs = OrderedDict(inputs)

        dim = sum(d.num_elements for d in inputs.values() if d.dtype == "real")
        if not get_tracing_state():
            assert dim
            assert len(prec_sqrt.shape) >= 2 and prec_sqrt.shape[-2] == dim
            rank = prec_sqrt.shape[-1]
            assert len(white_vec.shape) >= 1 and white_vec.shape[-1] == rank

        batch_shape = tuple(
            d.dtype for d in inputs.values() if isinstance(d.dtype, int)
        )
        if not get_tracing_state():
            assert prec_sqrt.shape[:-2] == batch_shape
            assert white_vec.shape[:-1] == batch_shape

        output = Real
        fresh = frozenset(inputs.keys())
        bound = {}
        super().__init__(inputs, output, fresh, bound)
        self.white_vec = white_vec
        self.prec_sqrt = prec_sqrt
        self.batch_shape = batch_shape
        self.event_shape = (dim,)

    @classmethod
    @contextmanager
    def set_compression_threshold(cls, threshold: float):
        pass

    def __repr__(self):
        return "Gaussian(..., ({}))".format(
            " ".join("({}, {}),".format(*kv) for kv in self.inputs.items())
        )

    @property
    def rank(self):
        pass

    @property
    def is_full_rank(self):
        pass

    @lazy_property
    def _precision(self):
        pass

    @lazy_property
    def _precision_chol(self):
        pass

    @lazy_property
    def _covariance(self):
        pass

    @lazy_property
    def _scale_tril(self):
        pass

    @lazy_property
    def _mean(self):
        pass

    @lazy_property
    def _info_vec(self):
        pass

    @lazy_property
    def _log_normalizer(self):
        pass

    @lazy_property
    def log_normalizer(self):
        pass

    def align(self, names):
        assert isinstance(names, tuple)
        assert all(name in self.inputs for name in names)
        if not names or names == tuple(self.inputs):
            return self

        inputs = OrderedDict((name, self.inputs[name]) for name in names)
        inputs.update(self.inputs)
        white_vec, prec_sqrt = align_gaussian(inputs, self)
        return Gaussian(white_vec, prec_sqrt, inputs)

    def eager_subs(self, subs):
        pass

    def _eager_subs_var(self, subs, remaining_subs):
        pass

    def _eager_subs_int(self, subs, remaining_subs):
        pass

    def _eager_subs_real(self, subs, remaining_subs):
        pass

    def _eager_subs_affine(self, subs, remaining_subs):
        pass

    def eager_reduce(self, op, reduced_vars):
        pass

    def _sample(self, sampled_vars, sample_inputs, rng_key):
        pass

    def _marginalize_after_split(
        self, inputs, int_inputs, prec_sqrt_a, prec_sqrt_b, precision_chol_a
    ):
        pass


def _sample_white_noise(sample_inputs, int_inputs, dim, prototype, rng_key):
    pass


@compress_gaussians.register(Gaussian, object, object, tuple)
def _compress_gaussians(white_vec, prec_sqrt, inputs):
    pass


@eager.register(Binary, AddOp, Gaussian, Gaussian)
def eager_add_gaussian_gaussian(op, lhs, rhs):
    pass


@eager.register(Binary, SubOp, Gaussian, Gaussian)
def eager_sub(op, lhs, rhs):
    pass


__all__ = [
    "BlockMatrix",
    "BlockVector",
    "Gaussian",
    "align_gaussian",
]
