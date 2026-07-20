
import numbers
import typing

import jax.numpy as np
import jax.random
import numpy as onp
from jax import lax
from jax.core import Tracer
from jax.scipy.linalg import cho_solve, solve_triangular
from jax.scipy.special import expit, gammaln, logsumexp

from .. import ops


array = (onp.generic, onp.ndarray, np.ndarray, Tracer)
ops.atanh.register(array)(np.arctanh)
ops.clamp.register(array)(np.clip)
ops.exp.register(array)(np.exp)
ops.log1p.register(array)(np.log1p)
ops.max.register(array, array)(np.maximum)
ops.min.register(array, array)(np.minimum)
ops.permute.register(array)(np.transpose)
ops.sigmoid.register(array)(expit)
ops.sqrt.register(array)(np.sqrt)
ops.tanh.register(array)(np.tanh)
ops.transpose.register(array)(np.swapaxes)
ops.flip.register(array)(np.flip)
ops.unsqueeze.register(array)(np.expand_dims)
ops.qr.register(array)(np.linalg.qr)




@ops.all.register(array)
def _all(x, axis, keepdims):
    pass


@ops.any.register(array)
def _any(x, axis, keepdims):
    pass


@ops.amax.register(array)
def _amax(x, axis, keepdims):
    pass


@ops.amin.register(array)
def _amin(x, axis, keepdims):
    pass


@ops.sum.register(array)
def _sum(x, axis, keepdims):
    pass


@ops.prod.register(array)
def _prod(x, axis, keepdims):
    pass


@ops.logsumexp.register(array)
def _logsumexp(x, axis, keepdims):
    pass


@ops.mean.register(array)
def _mean(x, axis, keepdims):
    pass


@ops.std.register(array)
def _std(x, axis, ddof, keepdims):
    pass


@ops.var.register(array)
def _var(x, axis, ddof, keepdims):
    pass




@ops.argmax.register(array)
def _argmax(x, axis, keepdims):
    pass


@ops.argmin.register(array)
def _argmin(x, axis, keepdims):
    pass


@ops.astype.register(array)
def _astype(x, dtype):
    pass


ops.cat.register(typing.Tuple[typing.Union[array], ...])(np.concatenate)


@ops.cholesky.register(array)
def _cholesky(x):
    pass


@ops.cholesky_inverse.register(array)
def _cholesky_inverse(x):
    pass


@ops.cholesky_solve.register(array, array)
def _cholesky_solve(x, y):
    pass


@ops.detach.register(array)
def _detach(x):
    pass


@ops.diagonal.register(array)
def _diagonal(x, dim1, dim2):
    pass


@ops.einsum.register(typing.Tuple[typing.Union[array], ...])
def _einsum(operands, equation):
    pass


@ops.expand.register(array)
def _expand(x, shape):
    pass


@ops.finfo.register(array)
def _finfo(x):
    pass


for typ in array:

    @ops.is_numeric_array.register(typ)
    def _is_numeric_array(x):
        pass


@ops.isnan.register(array)
def _isnan(x):
    pass


@ops.lgamma.register(array)
def _lgamma(x):
    pass


@ops.log.register(array)
def _log(x):
    pass


@ops.logaddexp.register(array, array)
def _safe_logaddexp_tensor_tensor(x, y):
    pass


@ops.logaddexp.register(numbers.Number, array)
def _safe_logaddexp_number_tensor(x, y):
    pass


@ops.logaddexp.register(array, numbers.Number)
def _safe_logaddexp_tensor_number(x, y):
    pass


ops.max.register(array, array)(np.maximum)
ops.min.register(array, array)(np.minimum)


@ops.max.register((int, float), array)
def _max(x, y):
    pass


@ops.max.register(array, (int, float))
def _max(x, y):
    pass


@ops.min.register((int, float), array)
def _min(x, y):
    pass


@ops.min.register(array, (int, float))
def _min(x, y):
    pass


@ops.new_full.register(array)
def _new_full(x, shape, value):
    pass


@ops.new_arange.register(array)
def _new_arange(x, start, stop, step):
    pass


@ops.new_eye.register(array)
def _new_eye(x, shape):
    pass


@ops.new_zeros.register(array)
def _new_zeros(x, shape):
    pass


@ops.randn.register(array)
def _randn(prototype, shape, rng_key=None):
    pass


@ops.reciprocal.register(array)
def _reciprocal(x):
    pass


@ops.safediv.register(array, array)
@ops.safediv.register((int, float), array)
def _safediv(x, y):
    pass


@ops.safesub.register(array, array)
@ops.safesub.register((int, float), array)
def _safesub(x, y):
    pass


@ops.scatter.register(array, tuple, array)
def _scatter(dest, indices, src):
    pass


@ops.stack.register(typing.Tuple[typing.Union[array + (int, float)], ...])
def _stack(parts, dim=0):
    pass


@ops.triangular_solve.register(array, array)
def _triangular_solve(x, y, upper=False, transpose=False):
    pass


@ops.triangular_inv.register(array)
def _triangular_inv(x, upper=False):
    pass
