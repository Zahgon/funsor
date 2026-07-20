
from funsor import ops

EINSUM_SYMBOLS_BASE = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


class Tensordot:

    def __init__(self, einsum):
        self.einsum = einsum

    def __call__(self, x, y, axes=2):
        xnd = len(x.shape)
        ynd = len(y.shape)

        if isinstance(axes, int):
            axes = range(xnd - axes, xnd), range(axes)

        if isinstance(axes[0], int):
            axes = (axes[0],), axes[1]
        if isinstance(axes[1], int):
            axes = axes[0], (axes[1],)

        x_ix = [None] * xnd
        y_ix = [None] * ynd
        out_ix = []

        available_ix = iter(EINSUM_SYMBOLS_BASE)
        for ax1, ax2 in zip(*axes):
            repeat = next(available_ix)
            x_ix[ax1] = repeat
            y_ix[ax2] = repeat

        for i in range(xnd):
            if x_ix[i] is None:
                leave = next(available_ix)
                x_ix[i] = leave
                out_ix.append(leave)
        for i in range(ynd):
            if y_ix[i] is None:
                leave = next(available_ix)
                y_ix[i] = leave
                out_ix.append(leave)

        einsum_str = "{},{}->{}".format(*map("".join, (x_ix, y_ix, out_ix)))
        return self.einsum(einsum_str, x, y)


def broadcast_all(*values, **kwargs):
    """
    Packed broadcasting of multiple tensors.
    """
    inputs = kwargs.get("inputs")
    dims = kwargs.get("dims")
    sizes = {
        dim: size
        for value, old_dims in zip(values, inputs)
        for dim, size in zip(old_dims, value.shape)
    }
    if dims is None:
        dims = "".join(sorted(sizes))
    else:
        assert set(dims) == set(sizes)
    shape = tuple(sizes[dim] for dim in dims)
    values = list(values)
    for i, (x, old_dims) in enumerate(zip(values, inputs)):
        if old_dims != dims:
            x = ops.permute(
                x, tuple(old_dims.index(dim) for dim in dims if dim in old_dims)
            )
            x = x.reshape(tuple(sizes[dim] if dim in old_dims else 1 for dim in dims))
            x = ops.expand(x, shape)
            assert len(x.shape) == len(dims)
            values[i] = x
    return tuple(values)
