
import numbers
import typing

import torch

import funsor.ops as ops


ops.abs.register(torch.Tensor)(torch.abs)
ops.atanh.register(torch.Tensor)(torch.atanh)
ops.cholesky_solve.register(torch.Tensor, torch.Tensor)(torch.cholesky_solve)
ops.clamp.register(torch.Tensor)(torch.clamp)
ops.exp.register(torch.Tensor)(torch.exp)
ops.full_like.register(torch.Tensor)(torch.full_like)
ops.log1p.register(torch.Tensor)(torch.log1p)
ops.sigmoid.register(torch.Tensor)(torch.sigmoid)
ops.sqrt.register(torch.Tensor)(torch.sqrt)
ops.tanh.register(torch.Tensor)(torch.tanh)
ops.transpose.register(torch.Tensor)(torch.transpose)
ops.flip.register(torch.Tensor)(torch.flip)
ops.unsqueeze.register(torch.Tensor)(torch.unsqueeze)
ops.qr.register(torch.Tensor)(torch.linalg.qr)




def _flatten_reduced_dim(x, dim):
    pass


@ops.all.register(torch.Tensor)
def _all(x, axis, keepdims):
    pass


@ops.any.register(torch.Tensor)
def _any(x, axis, keepdims):
    pass


@ops.amax.register(torch.Tensor)
def _amax(x, axis, keepdims):
    pass


@ops.amin.register(torch.Tensor)
def _amin(x, axis, keepdims):
    pass


@ops.sum.register(torch.Tensor)
def _sum(x, axis, keepdims):
    pass


@ops.prod.register(torch.Tensor)
def _prod(x, axis, keepdims):
    pass


@ops.logsumexp.register(torch.Tensor)
def _logsumexp(x, axis, keepdims):
    pass


@ops.mean.register(torch.Tensor)
def _mean(x, axis, keepdims):
    pass


@ops.std.register(torch.Tensor)
def _std(x, axis, ddof, keepdims):
    pass


@ops.var.register(torch.Tensor)
def _var(x, axis, ddof, keepdims):
    pass




@ops.argmax.register(torch.Tensor)
def _argmax(x, axis, keepdims):
    pass


@ops.argmin.register(torch.Tensor)
def _argmin(x, axis, keepdims):
    pass


@ops.astype.register(torch.Tensor)
def _astype(x, dtype):
    pass


ops.cat.register(typing.Tuple[torch.Tensor, ...])(torch.cat)


@ops.cholesky.register(torch.Tensor)
def _cholesky(x):
    pass


@ops.cholesky_inverse.register(torch.Tensor)
def _cholesky_inverse(x):
    pass


@ops.triangular_inv.register(torch.Tensor)
def _triangular_inv(x, upper=False):
    pass


@ops.detach.register(torch.Tensor)
def _detach(x):
    pass


@ops.diagonal.register(torch.Tensor)
def _diagonal(x, dim1, dim2):
    pass


@ops.einsum.register(typing.Tuple[torch.Tensor, ...])
def _einsum(operands, equation):
    pass


@ops.expand.register(torch.Tensor)
def _expand(x, shape):
    pass


@ops.finfo.register(torch.Tensor)
def _finfo(x):
    pass


@ops.is_numeric_array.register(torch.Tensor)
def _is_numeric_array(x):
    pass


@ops.isnan.register(torch.Tensor)
def _isnan(x):
    pass


@ops.lgamma.register(torch.Tensor)
def _lgamma(x):
    pass


@ops.log.register(torch.Tensor)
def _log(x):
    pass


@ops.logaddexp.register(torch.Tensor, torch.Tensor)
def _safe_logaddexp_tensor_tensor(x, y):
    pass


@ops.logaddexp.register(numbers.Number, torch.Tensor)
def _safe_logaddexp_number_tensor(x, y):
    pass


@ops.logaddexp.register(torch.Tensor, numbers.Number)
def _safe_logaddexp_tensor_number(x, y):
    pass


@ops.max.register(torch.Tensor, torch.Tensor)
def _max(x, y):
    pass


@ops.max.register(numbers.Number, torch.Tensor)
def _max(x, y):
    pass


@ops.max.register(torch.Tensor, numbers.Number)
def _max(x, y):
    pass


@ops.min.register(torch.Tensor, torch.Tensor)
def _min(x, y):
    pass


@ops.min.register(numbers.Number, torch.Tensor)
def _min(x, y):
    pass


@ops.min.register(torch.Tensor, numbers.Number)
def _min(x, y):
    pass


@ops.new_arange.register(torch.Tensor)
def _new_arange(x, start, stop, step):
    pass


@ops.new_eye.register(torch.Tensor)
def _new_eye(x, shape):
    pass


@ops.new_zeros.register(torch.Tensor)
def _new_zeros(x, shape):
    pass


@ops.new_full.register(torch.Tensor)
def _new_full(x, shape, value):
    pass


@ops.randn.register(torch.Tensor)
def _randn(prototype, shape, rng_key=None):
    pass


@ops.permute.register(torch.Tensor)
def _permute(x, dims):
    pass


@ops.pow.register(numbers.Number, torch.Tensor)
def _pow(x, y):
    pass


@ops.pow.register(torch.Tensor, numbers.Number)
@ops.pow.register(torch.Tensor, torch.Tensor)
def _pow(x, y):
    pass


@ops.reciprocal.register(torch.Tensor)
def _reciprocal(x):
    pass


@ops.safediv.register(torch.Tensor, torch.Tensor)
@ops.safediv.register(numbers.Number, torch.Tensor)
def _safediv(x, y):
    pass


@ops.safesub.register(torch.Tensor, torch.Tensor)
@ops.safesub.register(numbers.Number, torch.Tensor)
def _safesub(x, y):
    pass


@ops.scatter.register(torch.Tensor, tuple, torch.Tensor)
def _scatter(destin, indices, source):
    pass


@ops.scatter_add.register(torch.Tensor, tuple, torch.Tensor)
def _scatter_add(destin, indices, source):
    pass


ops.stack.register(typing.Tuple[torch.Tensor, ...])(torch.stack)


@ops.triangular_solve.register(torch.Tensor, torch.Tensor)
def _triangular_solve(x, y, upper=False, transpose=False):
    pass
