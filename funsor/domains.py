
import copyreg
import functools
import inspect
import operator
import warnings
from functools import reduce
from weakref import WeakValueDictionary

import funsor.ops as ops
from funsor.ops.builtin import parse_ellipsis, parse_slice
from funsor.util import broadcast_shape, get_backend, get_tracing_state, quote

Domain = type


class ArrayType(Domain):

    _type_cache = WeakValueDictionary()

    def __getitem__(cls, dtype_shape):
        dtype, shape = dtype_shape
        assert dtype is not None
        assert shape is not None

        if get_tracing_state() or get_backend() == "jax":
            if dtype not in (None, "real"):
                dtype = int(dtype)
            if shape is not None:
                shape = tuple(map(int, shape))

        assert cls.dtype in (None, dtype)
        assert cls.shape in (None, shape)
        key = dtype, shape
        result = ArrayType._type_cache.get(key, None)
        if result is None:
            if dtype == "real":
                assert all(isinstance(size, int) and size >= 0 for size in shape)
                name = (
                    "Reals[{}]".format(",".join(map(str, shape))) if shape else "Real"
                )
                result = RealsType(name, (), {"shape": shape})
            elif isinstance(dtype, int):
                assert dtype >= 0
                name = "Bint[{}]".format(",".join(map(str, (dtype,) + shape)))
                result = BintType(name, (), {"dtype": dtype, "shape": shape})
            else:
                raise ValueError("invalid dtype: {}".format(dtype))
            ArrayType._type_cache[key] = result
        return result

    def __subclasscheck__(cls, subcls):
        if not isinstance(subcls, ArrayType):
            return False
        if cls.dtype not in (None, subcls.dtype):
            return False
        if cls.shape not in (None, subcls.shape):
            return False
        return True

    def __repr__(cls):
        return cls.__name__

    def __str__(cls):
        return cls.__name__

    @property
    def num_elements(cls):
        pass

    @property
    def is_concrete(cls):
        pass


class BintType(ArrayType):
    def __getitem__(cls, size_shape):
        if isinstance(size_shape, tuple):
            size, shape = size_shape[0], size_shape[1:]
        else:
            size, shape = size_shape, ()
        return super().__getitem__((size, shape))

    def __subclasscheck__(cls, subcls):
        if not isinstance(subcls, BintType):
            return False
        if cls.dtype not in (None, subcls.dtype):
            return False
        if cls.shape not in (None, subcls.shape):
            return False
        return True

    @property
    def size(cls):
        return cls.dtype

    def __iter__(cls):
        from funsor.terms import Number

        return (Number(i, cls.size) for i in range(cls.size))


class RealsType(ArrayType):
    dtype = "real"

    def __getitem__(cls, shape):
        if not isinstance(shape, tuple):
            shape = (shape,)
        return super().__getitem__(("real", shape))

    def __subclasscheck__(cls, subcls):
        if not isinstance(subcls, RealsType):
            return False
        if cls.dtype not in (None, subcls.dtype):
            return False
        if cls.shape not in (None, subcls.shape):
            return False
        return True


def _pickle_array(cls):
    pass


copyreg.pickle(ArrayType, _pickle_array)
copyreg.pickle(BintType, _pickle_array)
copyreg.pickle(RealsType, _pickle_array)


class Array(metaclass=ArrayType):

    dtype = None
    shape = None


class Bint(metaclass=BintType):

    dtype = None
    shape = None


class Reals(metaclass=RealsType):

    shape = None


Real = Reals[()]


def reals(*args):
    warnings.warn(
        "reals(...) is deprecated, use Real or Reals[...] instead", DeprecationWarning
    )
    return Reals[args]


def bint(size):
    warnings.warn("bint(...) is deprecated, use Bint[...] instead", DeprecationWarning)
    return Bint[size]


class ProductDomain(Domain):
    _type_cache = WeakValueDictionary()

    def __getitem__(cls, arg_domains):
        try:
            return ProductDomain._type_cache[arg_domains]
        except KeyError:
            assert isinstance(arg_domains, tuple)
            assert all(isinstance(arg_domain, Domain) for arg_domain in arg_domains)
            subcls = type("Product_", (Product,), {"__args__": arg_domains})
            ProductDomain._type_cache[arg_domains] = subcls
            return subcls

    def __repr__(cls):
        return "Product[{}]".format(", ".join(map(repr, cls.__args__)))

    @property
    def __origin__(cls):
        return Product

    @property
    def shape(cls):
        return (len(cls.__args__),)


class Product(tuple, metaclass=ProductDomain):

    __args__ = NotImplemented


class DependentMeta(type):
    def __getitem__(cls, fn):
        return cls(fn)


class Dependent(metaclass=DependentMeta):

    def __init__(self, fn):
        function = type(lambda: None)
        self.fn = fn if isinstance(fn, function) else lambda: fn
        self.args = inspect.getfullargspec(fn)[0]

    def __call__(self, **kwargs):
        return self.fn(*map(kwargs.__getitem__, self.args))



quote.register_repr(BintType)
quote.register_repr(RealsType)


@functools.singledispatch
def find_domain(op, *domains):
    r"""
    Finds the :class:`Domain` resulting when applying ``op`` to ``domains``.
    :param callable op: An operation.
    :param Domain \*domains: One or more input domains.
    """
    raise NotImplementedError


@find_domain.register(ops.UnaryOp)
def _find_domain_pointwise_unary_generic(op, domain):
    pass


@find_domain.register(ops.AstypeOp)
def _find_domain_astype(op, domain):
    pass


@find_domain.register(ops.LogOp)
@find_domain.register(ops.ExpOp)
def _find_domain_log_exp(op, domain):
    pass


@find_domain.register(ops.ReductionOp)
def _find_domain_reduction(op, domain):
    pass


@find_domain.register(ops.ReshapeOp)
def _find_domain_reshape(op, domain):
    pass


@find_domain.register(ops.GetitemOp)
def _find_domain_getitem(op, lhs_domain, rhs_domain):
    pass


@find_domain.register(ops.GetsliceOp)
def _find_domain_getslice(op, domain):
    pass


@find_domain.register(ops.BinaryOp)
def _find_domain_pointwise_binary_generic(op, lhs, rhs):
    pass


@find_domain.register(ops.ComparisonOp)
def _find_domain_comparison(op, lhs, rhs):
    pass


@find_domain.register(ops.FloordivOp)
def _find_domain_floordiv(op, lhs, rhs):
    pass


@find_domain.register(ops.ModOp)
def _find_domain_mod(op, lhs, rhs):
    pass


@find_domain.register(ops.MatmulOp)
def _find_domain_matmul(op, lhs, rhs):
    pass


@find_domain.register(ops.AssociativeOp)
def _find_domain_associative_generic(op, *domains):
    pass


@find_domain.register(ops.WrappedTransformOp)
def _transform_find_domain(op, domain):
    pass


@find_domain.register(ops.LogAbsDetJacobianOp)
def _transform_log_abs_det_jacobian(op, domain, codomain):
    pass


@find_domain.register(ops.StackOp)
def _find_domain_stack(op, parts):
    pass


@find_domain.register(ops.CatOp)
def _find_domain_cat(op, parts):
    pass


@find_domain.register(ops.EinsumOp)
def _find_domain_einsum(op, operands):
    pass


__all__ = [
    "Bint",
    "BintType",
    "Dependent",
    "Domain",
    "Real",
    "Reals",
    "RealsType",
    "bint",
    "find_domain",
    "reals",
]
