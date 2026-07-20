
import functools
import itertools
import numbers
import typing
import warnings
from collections import OrderedDict, namedtuple
from functools import reduce, singledispatch
from weakref import WeakValueDictionary

from multipledispatch import dispatch

import funsor.interpreter as interpreter
import funsor.ops as ops
from funsor.domains import (
    Array,
    Bint,
    BintType,
    Domain,
    Product,
    ProductDomain,
    Real,
    find_domain,
)
from funsor.interpretations import (
    Interpretation,
    die,
    eager,
    lazy,
    moment_matching,
    reflect,
    sequential,
)
from funsor.interpreter import PatternMissingError, interpret
from funsor.ops import AssociativeOp, GetitemOp, Op
from funsor.ops.builtin import normalize_ellipsis, parse_ellipsis, parse_slice
from funsor.syntax import INFIX_OPERATORS, PREFIX_OPERATORS
from funsor.typing import GenericTypeMeta, Variadic, deep_type, get_args, get_origin
from funsor.util import getargspec, lazy_property, pretty, quote, register_pprint

from . import instrument, interpreter, ops

_PREFIX = {k: v for v, k, _ in PREFIX_OPERATORS}
_INFIX = {k: v for v, k, _ in INFIX_OPERATORS}


class SubstituteInterpretation(Interpretation):
    def __init__(self, subs, base_interpretation):
        super().__init__("subs")
        self.subs = subs
        self.base_interpretation = base_interpretation
        assert isinstance(subs, tuple)
        assert all(isinstance(v, Funsor) for k, v in subs)

    @property
    def is_total(self):
        pass

    def interpret(self, cls, *args):
        with self.base_interpretation:
            expr = cls(*args)
            fresh_subs = tuple((k, v) for k, v in self.subs if k in expr.fresh)
            if fresh_subs:
                expr = instrument.debug_logged(expr.eager_subs)(fresh_subs)
            if instrument.PROFILE:
                instrument.COUNTERS["interpretation"]["substitute"] += 1
            return expr


def substitute(expr, subs):
    if isinstance(subs, (dict, OrderedDict)):
        subs = tuple(subs.items())
    support = frozenset(k for k, v in subs)

    def stop(x):
        if interpreter.is_atom(x):
            return True
        if isinstance(x, Funsor) and support.isdisjoint(x.inputs):
            return True
        return False

    if stop(expr):
        return expr

    env = interpreter.anf(expr, stop)

    with SubstituteInterpretation(subs, interpreter.get_interpretation()):
        for key, value in env.items():
            args = tuple(
                c if interpreter.is_atom(c) else env.get(c, c)
                for c in interpreter.children(value)
            )
            if isinstance(value, (tuple, frozenset)):  # TODO absorb this into interpret
                env[key] = type(value)(args)
            else:
                env[key] = type(value)(*args)
    return env[expr]


def _alpha_mangle(expr):
    """
    Rename bound variables in expr to avoid conflict with any free variables.

    FIXME this does not avoid conflict with other bound variables.
    """
    alpha_subs = {
        name: interpreter.gensym(name + "__BOUND")
        for name in expr.bound
        if "__BOUND" not in name
    }
    if not alpha_subs:
        return expr

    ast_values = instrument.debug_logged(expr._alpha_convert)(alpha_subs)
    return reflect.interpret(type(expr), *ast_values)


@reflect.set_callable
def reflect(cls, *args, **kwargs):
    """
    Construct a funsor, populate ``._ast_values``, and cons hash.
    This is the only interpretation allowed to construct funsors.
    """
    if len(args) > len(cls._ast_fields):
        new_args = tuple(args[: len(cls._ast_fields) - 1]) + (
            args[len(cls._ast_fields) - 1 - len(args) :],
        )
        assert len(new_args) == len(cls._ast_fields)
        _, args = args, new_args

    cache_key = reflect.make_hash_key(cls, *args)
    if cache_key in cls._cons_cache:
        return cls._cons_cache[cache_key]

    arg_types = tuple(map(deep_type, args))
    cls_specific = get_origin(cls)[arg_types]
    result = super(FunsorMeta, cls_specific).__call__(*args)
    result._ast_values = args

    if instrument.PROFILE:
        size, depth, width = _get_ast_stats(result)
        instrument.COUNTERS["ast_size"][size] += 1
        instrument.COUNTERS["ast_depth"][depth] += 1
        classname = get_origin(cls).__name__
        instrument.COUNTERS["funsor"][classname] += 1
        instrument.COUNTERS[classname][width] += 1

    result = _alpha_mangle(result)

    cls._cons_cache[cache_key] = result
    return result


class FunsorMeta(GenericTypeMeta):

    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        register_pprint(cls)
        if not cls.__args__:
            cls._ast_fields = getargspec(cls.__init__)[0][1:]
            cls._cons_cache = WeakValueDictionary()

    def __getitem__(cls, arg_types):
        if not isinstance(arg_types, tuple):
            arg_types = (arg_types,)
        assert len(arg_types) == len(
            cls._ast_fields
        ), "Must provide exactly one type per subexpression"
        return super().__getitem__(arg_types)

    def __call__(cls, *args, **kwargs):
        if cls.__args__:
            cls = cls.__origin__

        if kwargs:
            args = list(args)
            for name in cls._ast_fields[len(args) :]:
                args.append(kwargs.pop(name))
            assert not kwargs, kwargs
            args = tuple(args)

        return interpret(cls, *args)

    @lazy_property
    def classname(cls):
        pass


def _convert_reduced_vars(reduced_vars, inputs):
    """
    Helper to convert the reduced_vars arg of ``.reduce()`` and friends.

    :param reduced_vars:
    :type reduced_vars: str, Variable, or set or frozenset thereof.
    :returns: A frozenset of reduced variables.
    :rtype: frozenset of :class:`Variable`
    """
    if isinstance(reduced_vars, frozenset):
        if all(isinstance(var, Variable) for var in reduced_vars):
            return reduced_vars

    if isinstance(reduced_vars, (str, Variable)):
        reduced_vars = {reduced_vars}
    assert isinstance(reduced_vars, (frozenset, set))
    assert all(isinstance(var, (str, Variable)) for var in reduced_vars)
    return frozenset(
        Variable(var, inputs[var]) if isinstance(var, str) else var
        for var in reduced_vars
    )


class Funsor(object, metaclass=FunsorMeta):

    def __init__(self, inputs, output, fresh=None, bound=None):
        fresh = frozenset() if fresh is None else fresh
        bound = {} if bound is None else bound
        assert isinstance(inputs, OrderedDict)
        for name, input_ in inputs.items():
            assert isinstance(name, str)
            assert isinstance(input_, Domain)
        assert isinstance(output, Domain)
        assert getattr(output, "is_concrete", True)
        assert isinstance(fresh, frozenset)
        assert isinstance(bound, dict)
        super(Funsor, self).__init__()
        self.inputs = inputs
        self.output = output
        self.fresh = fresh
        self.bound = bound

    @property
    def dtype(self):
        return self.output.dtype

    @property
    def shape(self):
        return self.output.shape

    @lazy_property
    def input_vars(self):
        pass

    def __copy__(self):
        return self

    def __reduce__(self):
        return type(self).__origin__, self._ast_values

    def __hash__(self):
        return id(self)

    @lazy_property
    def __annotations__(self):
        type_hints = dict(self.inputs)
        type_hints["return"] = self.output
        return type_hints

    def __repr__(self):
        try:
            ast_values = self._ast_values
        except AttributeError:
            return f"{type(self).__name__}(...)"
        return "{}({})".format(type(self).__name__, ", ".join(map(repr, ast_values)))

    def __str__(self):
        return "{}({})".format(
            type(self).__name__, ", ".join(map(str, self._ast_values))
        )

    def quote(self):
        return quote(self)

    def pretty(self, *args, **kwargs):
        return pretty(self, *args, **kwargs)

    def __contains__(self, item):
        raise TypeError

    def _alpha_convert(self, alpha_subs):
        """
        Rename bound variables while preserving all free variables.
        """
        assert set(alpha_subs).issubset(self.bound)
        return tuple(substitute(v, alpha_subs) for v in self._ast_values)

    def __call__(self, *args, **kwargs):
        """
        Partially evaluates this funsor by substituting dimensions.
        """
        subs = OrderedDict(zip(self.inputs, args))
        for k in self.inputs:
            if k in kwargs:
                subs[k] = kwargs[k]
        return Subs(self, tuple(subs.items()))

    def __bool__(self):
        if self.inputs or self.output.shape:
            raise ValueError(
                "bool value of Funsor with more than one value is ambiguous"
            )
        raise NotImplementedError

    def __nonzero__(self):
        return self.__bool__()

    def __len__(self):
        if not self.output.shape:
            raise ValueError("Funsor with empty shape has no len()")
        return self.output.shape[0]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def item(self):
        if self.inputs or self.output.shape:
            raise ValueError(
                "only one element Funsors can be converted to Python scalars"
            )
        raise NotImplementedError

    @property
    def requires_grad(self):
        pass

    def reduce(self, op, reduced_vars=None):
        """
        Reduce along all or a subset of inputs.

        :param op: A reduction operation.
        :type op: ~funsor.ops.AssociativeOp or ~funsor.ops.ReductionOp
        :param reduced_vars: An optional input name or set of names to reduce.
            If unspecified, all inputs will be reduced.
        :type reduced_vars: str, Variable, or set or frozenset thereof.
        """
        assert isinstance(op, (AssociativeOp, ops.ReductionOp))

        if reduced_vars is None:
            reduced_vars = frozenset(Variable(k, v) for k, v in self.inputs.items())
        else:
            reduced_vars = _convert_reduced_vars(reduced_vars, self.inputs)
        assert isinstance(reduced_vars, frozenset), reduced_vars

        if isinstance(op, ops.ReductionOp):
            if isinstance(op, ops.MeanOp):
                reduced_vars &= self.input_vars
                if not reduced_vars:
                    return self
                scale = 1 / reduce(ops.mul, [v.output.size for v in reduced_vars], 1)
                return self.reduce(ops.add, reduced_vars) * scale
            if isinstance(op, ops.VarOp):
                diff = self - self.reduce(ops.mean, reduced_vars)
                return (diff * diff).reduce(ops.mean, reduced_vars)
            if isinstance(op, ops.StdOp):
                return self.reduce(ops.var, reduced_vars).sqrt()
            raise NotImplementedError(f"Unsupported reduction op: {op}")
        assert isinstance(op, AssociativeOp)

        if not reduced_vars:
            return self
        return Reduce(op, self, reduced_vars)

    def approximate(self, op, guide, approx_vars=None):
        pass

    def sample(self, sampled_vars, sample_inputs=None, rng_key=None):
        """
        Create a Monte Carlo approximation to this funsor by replacing
        functions of ``sampled_vars`` with :class:`~funsor.delta.Delta` s.

        The result is a :class:`Funsor` with the same ``.inputs`` and
        ``.output`` as the original funsor (plus ``sample_inputs`` if
        provided), so that self can be replaced by the sample in expectation
        computations::

            y = x.sample(sampled_vars)
            assert y.inputs == x.inputs
            assert y.output == x.output
            exact = (x.exp() * integrand).reduce(ops.add)
            approx = (y.exp() * integrand).reduce(ops.add)

        If ``sample_inputs`` is provided, this creates a batch of samples.

        :param sampled_vars: A set of input variables to sample.
        :type sampled_vars: str, Variable, or set or frozenset thereof.
        :param OrderedDict sample_inputs: An optional mapping from variable
            name to :class:`~funsor.domains.Domain` over which samples will
            be batched.
        :param rng_key: a PRNG state to be used by JAX backend to generate
            random samples
        :type rng_key: None or JAX's random.PRNGKey
        """
        assert self.output == Real
        sampled_vars = _convert_reduced_vars(sampled_vars, self.inputs)
        sampled_vars = frozenset(v.name for v in sampled_vars)
        assert isinstance(sampled_vars, frozenset)
        if sample_inputs is None:
            sample_inputs = OrderedDict()
        assert isinstance(sample_inputs, OrderedDict)
        if sampled_vars.isdisjoint(self.inputs):
            return self

        result = instrument.debug_logged(self._sample)(
            sampled_vars, sample_inputs, rng_key
        )
        return result

    def _sample(self, sampled_vars, sample_inputs, rng_key):
        pass

    def align(self, names):
        """
        Align this funsor to match given ``names``.
        This is mainly useful in preparation for extracting ``.data``
        of a :class:`funsor.tensor.Tensor`.

        :param tuple names: A tuple of strings representing all names
            but in a new order.
        :return: A permuted funsor equivalent to self.
        :rtype: Funsor
        """
        assert isinstance(names, tuple)
        if not names or names == tuple(self.inputs):
            return self
        return Align(self, names)

    def eager_subs(self, subs):
        pass

    def eager_unary(self, op):
        pass

    def eager_reduce(self, op, reduced_vars):
        pass

    def sequential_reduce(self, op, reduced_vars):
        pass

    def moment_matching_reduce(self, op, reduced_vars):
        pass


    def __invert__(self):
        return Unary(ops.invert, self)

    def __pos__(self):
        return Unary(ops.pos, self)

    def __neg__(self):
        return Unary(ops.neg, self)

    def abs(self):
        return Unary(ops.abs, self)

    def atanh(self):
        pass

    def sqrt(self):
        return Unary(ops.sqrt, self)

    def exp(self):
        return Unary(ops.exp, self)

    def log(self):
        return Unary(ops.log, self)

    def log1p(self):
        pass

    def sigmoid(self):
        pass

    def tanh(self):
        pass

    def reshape(self, shape):
        return Unary(ops.ReshapeOp(shape), self)


    def all(self, axis=None, keepdims=False):
        pass

    def any(self, axis=None, keepdims=False):
        pass

    def argmax(self, axis=None, keepdims=False):
        pass

    def argmin(self, axis=None, keepdims=False):
        pass

    def max(self, axis=None, keepdims=False):
        pass

    def min(self, axis=None, keepdims=False):
        pass

    def sum(self, axis=None, keepdims=False):
        pass

    def prod(self, axis=None, keepdims=False):
        pass

    def logsumexp(self, axis=None, keepdims=False):
        return Unary(ops.LogsumexpOp(axis, keepdims), self)

    def mean(self, axis=None, keepdims=False):
        pass

    def std(self, axis=None, ddof=0, keepdims=False):
        pass

    def var(self, axis=None, ddof=0, keepdims=False):
        pass

    def __add__(self, other):
        return Binary(ops.add, self, to_funsor(other))

    def __radd__(self, other):
        return Binary(ops.add, self, to_funsor(other))

    def __sub__(self, other):
        return Binary(ops.sub, self, to_funsor(other))

    def __rsub__(self, other):
        return Binary(ops.sub, to_funsor(other), self)

    def __mul__(self, other):
        return Binary(ops.mul, self, to_funsor(other))

    def __rmul__(self, other):
        return Binary(ops.mul, self, to_funsor(other))

    def __truediv__(self, other):
        return Binary(ops.truediv, self, to_funsor(other))

    def __rtruediv__(self, other):
        return Binary(ops.truediv, to_funsor(other), self)

    def __floordiv__(self, other):
        return Binary(ops.floordiv, self, to_funsor(other))

    def __rfloordiv__(self, other):
        return Binary(ops.floordiv, to_funsor(other), self)

    def __matmul__(self, other):
        return Binary(ops.matmul, self, to_funsor(other))

    def __rmatmul__(self, other):
        return Binary(ops.matmul, to_funsor(other), self)

    def __mod__(self, other):
        return Binary(ops.mod, self, to_funsor(other))

    def __rmod__(self, other):
        return Binary(ops.mod, to_funsor(other), self)

    def __lshift__(self, other):
        return Binary(ops.lshift, self, to_funsor(other))

    def __rlshift__(self, other):
        return Binary(ops.lshift, to_funsor(other), self)

    def __rshift__(self, other):
        return Binary(ops.rshift, self, to_funsor(other))

    def __rrshift__(self, other):
        return Binary(ops.rshift, to_funsor(other), self)

    def __pow__(self, other):
        return Binary(ops.pow, self, to_funsor(other))

    def __rpow__(self, other):
        return Binary(ops.pow, to_funsor(other), self)

    def __and__(self, other):
        return Binary(ops.and_, self, to_funsor(other))

    def __rand__(self, other):
        return Binary(ops.and_, self, to_funsor(other))

    def __or__(self, other):
        return Binary(ops.or_, self, to_funsor(other))

    def __ror__(self, other):
        return Binary(ops.or_, self, to_funsor(other))

    def __xor__(self, other):
        return Binary(ops.xor, self, to_funsor(other))

    def __eq__(self, other):
        return Binary(ops.eq, self, to_funsor(other))

    def __ne__(self, other):
        return Binary(ops.ne, self, to_funsor(other))

    def __lt__(self, other):
        return Binary(ops.lt, self, to_funsor(other))

    def __le__(self, other):
        return Binary(ops.le, self, to_funsor(other))

    def __gt__(self, other):
        return Binary(ops.gt, self, to_funsor(other))

    def __ge__(self, other):
        return Binary(ops.ge, self, to_funsor(other))

    def __getitem__(self, other):
        """
        Helper to desugar into either ops.getitem (for advanced indexing
        involving Funsors as indices) or ops.getslice (for simple indexing
        involving only integers, slices, None, and Ellipsis).
        """
        if type(other) is not tuple:
            if isinstance(other, ops.getslice.supported_types):
                return ops.getslice(self, other)
            other = to_funsor(other, Bint[self.output.shape[0]])
            return Binary(ops.getitem, self, other)

        if all(isinstance(part, ops.getslice.supported_types) for part in other):
            return ops.getslice(self, other)

        if any(part is Ellipsis for part in other):
            left, right = parse_ellipsis(other)
            missing = len(self.output.shape) - len(left) - len(right)
            assert missing >= 0
            middle = [slice(None)] * missing
            other = tuple(left + middle + right)

        result = self
        offset = 0
        for part in other:
            if part is None:
                raise NotImplementedError("TODO")
            if isinstance(part, slice):
                if part != slice(None):
                    raise NotImplementedError("TODO support nontrivial slicing")
                offset += 1
            else:
                part = to_funsor(part, Bint[result.output.shape[offset]])
                result = Binary(GetitemOp(offset), result, part)
        return result


@quote.register(Funsor)
def _(arg, indent, out):
    pass


interpreter.children.register(Funsor)(interpreter.children_funsor)


@singledispatch
def to_funsor(x, output=None, dim_to_name=None, **kwargs):
    """
    Convert to a :class:`Funsor` .
    Only :class:`Funsor` s and scalars are accepted.

    :param x: An object.
    :param funsor.domains.Domain output: An optional output hint.
    :param OrderedDict dim_to_name: An optional mapping from negative batch dimensions to name strings.
    :return: A Funsor equivalent to ``x``.
    :rtype: Funsor
    :raises: ValueError
    """
    raise ValueError("Cannot convert to Funsor: {}".format(repr(x)))


@to_funsor.register(Funsor)
def funsor_to_funsor(x, output=None, dim_to_name=None):
    pass


@singledispatch
def to_data(x, name_to_dim=None, **kwargs):
    """
    Extract a python object from a :class:`Funsor`.

    Raises a ``ValueError`` if free variables remain or if the funsor is lazy.

    :param x: An object, possibly a :class:`Funsor`.
    :param OrderedDict name_to_dim: An optional inputs hint.
    :return: A non-funsor equivalent to ``x``.
    :raises: ValueError if any free variables remain.
    :raises: PatternMissingError if funsor is not fully evaluated.
    """
    return x


@to_data.register(Funsor)
def _to_data_funsor(x, name_to_dim=None):
    pass


class Variable(Funsor):

    def __init__(self, name, output):
        inputs = OrderedDict([(name, output)])
        fresh = frozenset({name})
        super(Variable, self).__init__(inputs, output, fresh)
        self.name = name

    def __repr__(self):
        return "Variable({}, {})".format(repr(self.name), repr(self.output))

    def __str__(self):
        return self.name

    def eager_subs(self, subs):
        pass


@to_funsor.register(str)
def name_to_funsor(name, output=None):
    pass


class SubsMeta(FunsorMeta):

    def __call__(cls, arg, subs):
        subs = tuple(
            (k, to_funsor(v, arg.inputs[k])) for k, v in subs if k in arg.inputs
        )
        return super().__call__(arg, subs)


class Subs(Funsor, metaclass=SubsMeta):

    def __init__(self, arg, subs):
        assert isinstance(arg, Funsor)
        assert isinstance(subs, tuple)
        for key, value in subs:
            assert isinstance(key, str)
            assert key in arg.inputs
            assert isinstance(value, Funsor)
        inputs = arg.inputs.copy()
        for key, value in subs:
            del inputs[key]
        for key, value in subs:
            inputs.update(value.inputs)
        fresh = frozenset()
        bound = {key: value.output for key, value in subs}
        super(Subs, self).__init__(inputs, arg.output, fresh, bound)
        self.arg = arg
        self.subs = OrderedDict(subs)

    def __repr__(self):
        return "{}({})".format(
            repr(self.arg), ", ".join(f"{k}={repr(v)}" for k, v in self.subs.items())
        )

    def __str__(self):
        return "{}({})".format(
            str(self.arg), ", ".join(f"{k}={str(v)}" for k, v in self.subs.items())
        )

    def _alpha_convert(self, alpha_subs):
        assert set(alpha_subs).issubset(self.bound)
        alpha_subs = {
            k: to_funsor(v, self.subs[k].output) for k, v in alpha_subs.items()
        }
        arg, subs = self._ast_values
        arg = substitute(arg, alpha_subs)
        subs = tuple((str(alpha_subs.get(k, k)), v) for k, v in subs)
        return arg, subs

    def _sample(self, sampled_vars, sample_inputs, rng_key=None):
        pass


@lazy.register(Subs, Funsor, object)
@eager.register(Subs, Funsor, object)
def eager_subs_funsor(arg, subs):
    pass


@lazy.register(Subs, Subs, object)
@eager.register(Subs, Subs, object)
def eager_subs_subs(arg, subs):
    pass


@die.register(Subs, Funsor, tuple)
def die_subs(arg, subs):
    pass


class Unary(Funsor):

    def __init__(self, op, arg):
        assert callable(op)
        assert isinstance(arg, Funsor)
        output = find_domain(op, arg.output)
        super(Unary, self).__init__(arg.inputs, output)
        self.op = op
        self.arg = arg

    def __repr__(self):
        if self.op in _PREFIX:
            return "({}{})".format(_PREFIX[self.op], repr(self.arg))
        return super().__repr__()

    def __str__(self):
        if self.op in _PREFIX:
            return "({}{})".format(_PREFIX[self.op], str(self.arg))
        return super().__str__()


@eager.register(Unary, Op, Funsor)
def eager_unary(op, arg):
    pass


@eager.register(Unary, AssociativeOp, Funsor)
def eager_unary(op, arg):
    pass


@die.register(Unary, Op, Funsor)
def die_unary(op, arg):
    pass


class Binary(Funsor):

    def __init__(self, op, lhs, rhs):
        assert callable(op)
        assert isinstance(lhs, Funsor)
        assert isinstance(rhs, Funsor)
        inputs = lhs.inputs.copy()
        inputs.update(rhs.inputs)
        output = find_domain(op, lhs.output, rhs.output)
        super(Binary, self).__init__(inputs, output)
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        if self.op in _INFIX:
            return "({} {} {})".format(repr(self.lhs), _INFIX[self.op], repr(self.rhs))
        return super().__repr__()

    def __str__(self):
        if self.op in _INFIX:
            return "({} {} {})".format(str(self.lhs), _INFIX[self.op], str(self.rhs))
        return super().__str__()


@die.register(Binary, Op, Funsor, Funsor)
def die_binary(op, lhs, rhs):
    pass


class Reduce(Funsor):

    def __init__(self, op, arg, reduced_vars):
        assert isinstance(op, AssociativeOp)
        assert isinstance(arg, Funsor)
        assert isinstance(reduced_vars, frozenset)
        assert all(isinstance(v, Variable) for v in reduced_vars)
        reduced_names = frozenset(v.name for v in reduced_vars)
        inputs = OrderedDict(
            (k, v) for k, v in arg.inputs.items() if k not in reduced_names
        )
        output = arg.output
        fresh = frozenset()
        bound = {var.name: var.output for var in reduced_vars}
        super(Reduce, self).__init__(inputs, output, fresh, bound)
        self.op = op
        self.arg = arg
        self.reduced_vars = reduced_vars

    def __repr__(self):
        assert self.reduced_vars
        if self.reduced_vars == self.arg.input_vars:
            return f"{repr(self.arg)}.reduce({self.op.__name__})"
        rvars = [
            f'"{v.name}"' if v in self.arg.input_vars else repr(v)
            for v in self.reduced_vars
        ]
        return "{}.reduce({}, {{{}}})".format(
            repr(self.arg), self.op.__name__, ", ".join(rvars)
        )

    def __str__(self):
        assert self.reduced_vars
        if self.reduced_vars == self.arg.input_vars:
            return f"{str(self.arg)}.reduce({self.op.__name__})"
        rvars = [
            f'"{v.name}"' if v in self.arg.input_vars else repr(v)
            for v in self.reduced_vars
        ]
        return "{}.reduce({}, {{{}}})".format(
            str(self.arg), self.op.__name__, ", ".join(rvars)
        )

    def _alpha_convert(self, alpha_subs):
        alpha_subs = {
            k: to_funsor(v, self.arg.inputs[k]) for k, v in alpha_subs.items()
        }
        op, arg, reduced_vars = super()._alpha_convert(alpha_subs)
        reduced_vars = frozenset(alpha_subs.get(var.name, var) for var in reduced_vars)
        return op, arg, reduced_vars


def _reduce_unrelated_vars(op, arg, reduced_vars):
    pass


@lazy.register(Reduce, AssociativeOp, Funsor, frozenset)
def lazy_reduce(op, arg, reduced_vars):
    pass


@eager.register(Reduce, AssociativeOp, Funsor, frozenset)
def eager_reduce(op, arg, reduced_vars):
    pass


@sequential.register(Reduce, AssociativeOp, Funsor, frozenset)
def sequential_reduce(op, arg, reduced_vars):
    pass


@moment_matching.register(Reduce, AssociativeOp, Funsor, frozenset)
def moment_matching_reduce(op, arg, reduced_vars):
    pass


@die.register(Reduce, Op, Funsor, frozenset)
def die_reduce(op, arg, reduced_vars):
    pass


class Scatter(Funsor):

    def __init__(self, op, subs, source, reduced_vars):
        assert isinstance(op, AssociativeOp)
        assert isinstance(subs, tuple)
        assert len(subs) == len(set(key for key, value in subs))
        assert isinstance(source, Funsor)
        assert isinstance(reduced_vars, frozenset)
        assert all(isinstance(v, Variable) for v in reduced_vars)
        reduced_names = frozenset(v.name for v in reduced_vars)

        inputs = OrderedDict()
        for key, value in subs:
            assert isinstance(key, str)
            assert isinstance(value, Funsor)
            assert key not in source.inputs
            assert key not in reduced_names
            for k, d in value.inputs.items():
                d2 = inputs.setdefault(k, d)
                assert d2 == d
        for k, d in source.inputs.items():
            d2 = inputs.setdefault(k, d)
            assert d2 == d
        for key, value in subs:
            assert key not in inputs
            inputs[key] = value.output

        inputs = OrderedDict(
            (k, d) for k, d in inputs.items() if k not in reduced_names
        )
        fresh = frozenset(key for key, value in subs)
        bound = {v.name: v.output for v in reduced_vars}
        super().__init__(inputs, source.output, fresh, bound)
        self.op = op
        self.subs = subs
        self.source = source
        self.reduced_vars = reduced_vars

    def _alpha_convert(self, alpha_subs):
        alpha_subs = {k: to_funsor(v, self.bound[k]) for k, v in alpha_subs.items()}
        op, subs, source, reduced_vars = super()._alpha_convert(alpha_subs)
        reduced_vars = frozenset(alpha_subs.get(var.name, var) for var in reduced_vars)
        return op, subs, source, reduced_vars

    def eager_subs(self, subs):
        pass


class Approximate(Funsor):

    def __init__(self, op, model, guide, approx_vars):
        assert isinstance(op, AssociativeOp)
        assert isinstance(model, Funsor)
        assert isinstance(guide, Funsor)
        assert model.output is guide.output
        assert isinstance(approx_vars, frozenset), approx_vars
        inputs = model.inputs.copy()
        inputs.update(guide.inputs)
        output = model.output
        fresh = frozenset(v.name for v in approx_vars)
        bound = {v.name: v.output for v in approx_vars}
        super().__init__(inputs, output, fresh, bound)
        self.op = op
        self.model = model
        self.guide = guide
        self.approx_vars = approx_vars

    def _alpha_convert(self, alpha_subs):
        alpha_subs = {k: to_funsor(v, self.bound[k]) for k, v in alpha_subs.items()}
        op, model, guide, approx_vars = super()._alpha_convert(alpha_subs)
        approx_vars = frozenset(alpha_subs.get(var.name, var) for var in approx_vars)
        return op, model, guide, approx_vars


@eager.register(Approximate, AssociativeOp, Funsor, Funsor, frozenset)
def eager_approximate(op, model, guide, approx_vars):
    pass


class NumberMeta(FunsorMeta):

    def __call__(cls, data, dtype=None):
        if dtype is None:
            dtype = "real"
        return super(NumberMeta, cls).__call__(data, dtype)


class Number(Funsor, metaclass=NumberMeta):

    def __init__(self, data, dtype=None):
        assert isinstance(data, numbers.Number)
        if isinstance(dtype, int):
            data = type(dtype)(data)
            if dtype != 2:  # booleans have bitwise interpretation
                assert 0 <= data and data < dtype
        else:
            assert isinstance(dtype, str) and dtype == "real"
            data = float(data)
        inputs = OrderedDict()
        output = Array[dtype, ()]
        super(Number, self).__init__(inputs, output)
        self.data = data

    def __repr__(self):
        if self.dtype == "real":
            return f"Number({str(self.data)})"
        else:
            return f"Number({str(self.data)}, {self.dtype})"

    def __str__(self):
        return str(self.data)

    def __int__(self):
        return int(self.data)

    def __float__(self):
        return float(self.data)

    def __bool__(self):
        return bool(self.data)

    def item(self):
        return self.data

    def eager_unary(self, op):
        pass


@to_funsor.register(numbers.Number)
def number_to_funsor(x, output=None, dim_to_name=None):
    pass


@to_data.register(Number)
def _to_data_number(x, name_to_dim=None):
    pass


@eager.register(Binary, Op, Number, Number)
def eager_binary_number_number(op, lhs, rhs):
    pass


class SliceMeta(FunsorMeta):

    def __call__(cls, name, *args, **kwargs):
        start = 0
        step = 1
        dtype = None
        if len(args) == 1:
            stop = args[0]
            dtype = kwargs.pop("dtype", stop)
        elif len(args) == 2:
            start, stop = args
            dtype = kwargs.pop("dtype", stop)
        elif len(args) == 3:
            start, stop, step = args
            dtype = kwargs.pop("dtype", stop)
        elif len(args) == 4:
            start, stop, step, dtype = args
        else:
            raise ValueError
        if step <= 0:
            raise ValueError
        stop = min(dtype, max(start, stop))
        return super().__call__(name, start, stop, step, dtype)


class Slice(Funsor, metaclass=SliceMeta):

    def __init__(self, name, start, stop, step, dtype):
        assert isinstance(name, str)
        assert isinstance(start, int) and start >= 0
        assert isinstance(stop, int) and stop >= start
        assert isinstance(step, int) and step > 0
        assert isinstance(dtype, int)
        size = max(0, (stop + step - 1 - start) // step)
        inputs = OrderedDict([(name, Bint[size])])
        output = Bint[dtype]
        fresh = frozenset({name})
        super().__init__(inputs, output, fresh)
        self.name = name
        self.slice = slice(start, stop, step)

    def eager_subs(self, subs):
        pass


@to_funsor.register(slice)
def slice_to_funsor(s, output=None, dim_to_name=None):
    pass


class Align(Funsor):

    def __init__(self, arg, names):
        assert isinstance(arg, Funsor)
        assert isinstance(names, tuple)
        assert all(isinstance(name, str) for name in names)
        assert all(name in arg.inputs for name in names)
        inputs = OrderedDict((name, arg.inputs[name]) for name in names)
        inputs.update(arg.inputs)
        output = arg.output
        fresh = frozenset()  # TODO get this right
        bound = {}
        super(Align, self).__init__(inputs, output, fresh, bound)
        self.arg = arg

    def align(self, names):
        return self.arg.align(names)

    def eager_unary(self, op):
        pass

    def eager_reduce(self, op, reduced_vars):
        pass


@eager.register(Align, Funsor, tuple)
def eager_align(arg, names):
    pass


@eager.register(Binary, Op, Align, Funsor)
def eager_binary_align_funsor(op, lhs, rhs):
    pass


@eager.register(Binary, Op, Funsor, Align)
def eager_binary_funsor_align(op, lhs, rhs):
    pass


@eager.register(Binary, Op, Align, Align)
def eager_binary_align_align(op, lhs, rhs):
    pass


class Finitary(Funsor):
    def __init__(self, op, args):
        assert isinstance(op, ops.Op)
        assert isinstance(args, tuple)
        assert all(isinstance(v, Funsor) for v in args)
        inputs = OrderedDict()
        for arg in args:
            inputs.update(arg.inputs)
        output = find_domain(op, tuple(arg.output for arg in args))
        super().__init__(inputs, output)
        self.op = op
        self.args = args


class Stack(Funsor):

    def __init__(self, name, parts):
        assert isinstance(name, str)
        assert isinstance(parts, tuple)
        assert parts
        assert not any(name in x.inputs for x in parts)
        assert len(set(x.output for x in parts)) == 1
        output = parts[0].output
        domain = Bint[len(parts)]
        inputs = OrderedDict([(name, domain)])
        for x in parts:
            inputs.update(x.inputs)
        fresh = frozenset({name})
        super().__init__(inputs, output, fresh)
        self.name = name
        self.parts = parts

    def eager_subs(self, subs):
        pass

    def eager_reduce(self, op, reduced_vars):
        pass


@eager.register(Stack, str, tuple)
def eager_stack(name, parts):
    pass


@dispatch(str, Variadic[Funsor])
def eager_stack_homogeneous(name, *parts):
    pass


class CatMeta(FunsorMeta):

    def __call__(cls, name, parts, part_name=None):
        if part_name is None:
            part_name = name
        return super().__call__(name, parts, part_name)


class Cat(Funsor, metaclass=CatMeta):

    def __init__(self, name, parts, part_name=None):
        assert isinstance(name, str)
        assert isinstance(parts, tuple)
        assert isinstance(part_name, str)
        assert parts
        for part in parts:
            assert part_name in part.inputs, (part_name, part.inputs)
        if part_name != name:
            assert not any(name in x.inputs for x in parts)
        assert len(set(x.output for x in parts)) == 1
        output = parts[0].output
        inputs = OrderedDict()
        for x in parts:
            inputs.update(x.inputs)
        del inputs[part_name]
        inputs[name] = Bint[sum(x.inputs[part_name].size for x in parts)]
        fresh = frozenset({name})
        bound = {part_name: x.inputs[part_name]}
        super().__init__(inputs, output, fresh, bound)
        self.name = name
        self.parts = parts
        self.part_name = part_name

    def _alpha_convert(self, alpha_subs):
        assert len(alpha_subs) == 1
        part_name = alpha_subs[self.part_name]
        parts = tuple(
            substitute(
                p, {self.part_name: to_funsor(part_name, p.inputs[self.part_name])}
            )
            for p in self.parts
        )
        return self.name, parts, part_name

    def eager_subs(self, subs):
        pass


@eager.register(Cat, str, tuple, str)
def eager_cat(name, parts, part_name):
    pass


@dispatch(str, str, Variadic[Funsor])
def eager_cat_homogeneous(name, part_name, *parts):
    pass


class Lambda(Funsor):

    def __init__(self, var, expr):
        assert isinstance(var, Variable)
        assert isinstance(var.dtype, int)
        assert isinstance(expr, Funsor)
        inputs = expr.inputs.copy()
        inputs.pop(var.name, None)
        shape = (var.dtype,) + expr.output.shape
        output = Array[expr.dtype, shape]
        fresh = frozenset()
        bound = {var.name: var.output}
        super(Lambda, self).__init__(inputs, output, fresh, bound)
        self.var = var
        self.expr = expr

    def _alpha_convert(self, alpha_subs):
        alpha_subs = {
            k: to_funsor(v, self.var.inputs[k]) for k, v in alpha_subs.items()
        }
        return super()._alpha_convert(alpha_subs)


@eager.register(Binary, GetitemOp, Lambda, (Funsor, Align))
def eager_getitem_lambda(op, lhs, rhs):
    pass


@eager.register(Unary, ops.GetsliceOp, Lambda)
def eager_getslice_lambda(op, x):
    pass


class Independent(Funsor):

    def __init__(self, fn, reals_var, bint_var, diag_var):
        assert isinstance(fn, Funsor)
        assert isinstance(reals_var, str)
        assert isinstance(bint_var, str)
        assert bint_var in fn.inputs, (bint_var, fn.inputs)
        assert isinstance(fn.inputs[bint_var].dtype, int)
        assert isinstance(diag_var, str)
        assert diag_var in fn.inputs
        inputs = fn.inputs.copy()
        diag_input = inputs.pop(diag_var)
        shape = (inputs.pop(bint_var).dtype,) + diag_input.shape
        assert reals_var not in inputs
        inputs[reals_var] = Array[diag_input.dtype, shape]
        fresh = frozenset({reals_var})
        bound = {bint_var: fn.inputs[bint_var], diag_var: fn.inputs[diag_var]}
        super(Independent, self).__init__(inputs, fn.output, fresh, bound)
        self.fn = fn
        self.reals_var = reals_var
        self.bint_var = bint_var
        self.diag_var = diag_var

    def _alpha_convert(self, alpha_subs):
        alpha_subs = {k: to_funsor(v, self.fn.inputs[k]) for k, v in alpha_subs.items()}
        fn, reals_var, bint_var, diag_var = super()._alpha_convert(alpha_subs)
        bint_var = str(alpha_subs.get(bint_var, bint_var))
        diag_var = str(alpha_subs.get(diag_var, diag_var))
        return fn, reals_var, bint_var, diag_var

    def _sample(self, sampled_vars, sample_inputs, rng_key=None):
        pass

    def eager_subs(self, subs):
        pass

    def mean(self):
        raise NotImplementedError("mean() not yet implemented for Independent")

    def variance(self):
        raise NotImplementedError("variance() not yet implemented for Independent")

    def entropy(self):
        raise NotImplementedError("entropy() not yet implemented for Independent")


@eager.register(Independent, Funsor, str, str, str)
def eager_independent_trivial(fn, reals_var, bint_var, diag_var):
    pass


class Tuple(Funsor):

    def __init__(self, args):
        assert isinstance(args, tuple)
        assert all(isinstance(arg, Funsor) for arg in args)
        inputs = OrderedDict()
        for arg in args:
            inputs.update(arg.inputs)
        output = Product[tuple(arg.output for arg in args)]
        super().__init__(inputs, output)
        self.args = args

    def __iter__(self):
        for i in range(len(self.args)):
            yield self[i]


@to_funsor.register(tuple)
def tuple_to_funsor(args, output=None, dim_to_name=None):
    pass


@lazy.register(Binary, GetitemOp, Tuple, Number)
@eager.register(Binary, GetitemOp, Tuple, Number)
def eager_getitem_tuple(op, lhs, rhs):
    pass


@lazy.register(Unary, ops.GetsliceOp, Tuple)
@eager.register(Unary, ops.GetsliceOp, Tuple)
def eager_getslice_tuple(op, x):
    pass


def _symbolic(inputs, output, fn):
    args, vargs, kwargs, defaults = getargspec(fn)
    assert not vargs
    assert not kwargs
    names = tuple(args)
    if isinstance(inputs, dict):
        args = tuple(Variable(name, inputs[name]) for name in names if name in inputs)
    else:
        args = tuple(Variable(name, domain) for (name, domain) in zip(names, inputs))
    assert len(args) == len(inputs)
    return to_funsor(fn(*args), output).align(names)


def symbolic(*signature):
    r"""
    Decorator to construct a symbolic :class:`Funsor` with one free
    :class:`Variable` per function arg. This can be used either with explicit
    types or with type hints::

        # Using type hints:
        @symbolic
        def xpyi(x: Real, y: Reals[3], i: Bint[3]):
            return x + y[i]

        # Using explicit type annotations:
        @symbolic(Real, Reals[3], Bint[3])
        def xpyi(x: Real, y: Reals[3], i: Bint[3]):
            return x + y[i]

    :param \*signature: A sequence if input domains.
    """
    if len(signature) == 1:
        fn = signature[0]
        if callable(fn) and not isinstance(fn, Domain):
            inputs = typing.get_type_hints(fn)
            output = inputs.pop("return", None)
            return _symbolic(inputs, output, fn)
    output = None
    return functools.partial(_symbolic, inputs, output)


def of_shape(*shape):
    warnings.warn("@of_shape is deprecated, use @symbolic instead", DeprecationWarning)
    return symbolic(*shape)


AstStats = namedtuple("AstStats", ("size", "depth", "width"))


@singledispatch
def _count_funsors(x):
    pass


@_count_funsors.register(Funsor)
def _(x):
    pass


@_count_funsors.register(tuple)
def _(x):
    pass


@singledispatch
def _get_ast_stats(x):
    return AstStats(1, 1, 0)


@_get_ast_stats.register(Funsor)
def _(x):
    pass


@_get_ast_stats.register(tuple)
def _(x):
    pass




@quote.register(Variable)
@quote.register(Number)
@quote.register(Slice)
def quote_inplace_oneline(arg, indent, out):
    pass


@quote.register(Unary)
@quote.register(Binary)
@quote.register(Reduce)
@quote.register(Stack)
@quote.register(Cat)
@quote.register(Lambda)
def quote_inplace_first_arg_on_first_line(arg, indent, out):
    pass


@ops.UnaryOp.subclass_register(Funsor)
def unary_funsor(cls, arg, *args, **kwargs):
    pass


@ops.BinaryOp.subclass_register(Funsor, Funsor)
def binary_funsor_funsor(cls, lhs, rhs, *args, **kwargs):
    pass


@ops.BinaryOp.subclass_register(object, Funsor)
def binary_object_funsor(cls, lhs, rhs, *args, **kwargs):
    pass


@ops.BinaryOp.subclass_register(Funsor, object)
def binary_funsor_object(cls, lhs, rhs, *args, **kwargs):
    pass


@ops.TernaryOp.subclass_register(Funsor, Funsor, Funsor)
@ops.TernaryOp.subclass_register(Funsor, Funsor, object)
@ops.TernaryOp.subclass_register(Funsor, object, object)
@ops.TernaryOp.subclass_register(object, Funsor, object)
@ops.TernaryOp.subclass_register(object, object, Funsor)
def ternary_funsor_object(cls, x, y, z, *args, **kwargs):
    pass


@ops.FinitaryOp.subclass_register(typing.Tuple[Funsor, ...])
def finitary_funsor(cls, arg, *args, **kwargs):
    pass


__all__ = [
    "Approximate",
    "Binary",
    "Cat",
    "Funsor",
    "Independent",
    "Lambda",
    "Number",
    "Reduce",
    "Scatter",
    "Stack",
    "Slice",
    "Subs",
    "Unary",
    "Variable",
    "of_shape",
    "to_data",
    "to_funsor",
]
