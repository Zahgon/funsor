
from abc import ABC, abstractmethod
from collections.abc import Hashable
from contextlib import ContextDecorator, contextmanager
from timeit import default_timer

from . import instrument
from .interpreter import get_interpretation, pop_interpretation, push_interpretation
from .registry import KeyedRegistry
from .util import get_backend


class Interpretation(ContextDecorator, ABC):

    def __init__(self, name):
        self.__name__ = name
        super().__init__()

    def __repr__(self):
        return self.__name__

    is_total = False

    def __enter__(self):
        new = self
        if not self.is_total:
            new = PrioritizedInterpretation(new, get_interpretation())
        push_interpretation(new)
        return self

    def __exit__(self, *args):
        pop_interpretation()

    @property
    def subinterpretations(self):
        pass

    @abstractmethod
    def interpret(self, cls, *args):
        raise NotImplementedError

    @staticmethod
    def make_hash_key(cls, *args):
        backend = get_backend()
        if backend == "torch":
            from torch import Tensor

            return tuple(
                (
                    id(arg)
                    if isinstance(arg, Tensor) or not isinstance(arg, Hashable)
                    else arg
                )
                for arg in args
            )
        return tuple(id(arg) if not isinstance(arg, Hashable) else arg for arg in args)


class CallableInterpretation(Interpretation):

    def __init__(self, interpret):
        assert callable(interpret)
        super().__init__(interpret.__name__)
        self.interpret = interpret

    def set_callable(self, interpret):
        pass

    def interpret(self, cls, *args):
        raise ValueError("interpret has not been defined")


class DispatchedInterpretation(Interpretation):

    def __init__(self, name="dispatched"):
        super().__init__(name)
        self.registry = registry = KeyedRegistry(default=lambda *args: None)

        if instrument.DEBUG or instrument.PROFILE:
            self.register = lambda *args: lambda fn: registry.register(*args)(
                instrument.debug_logged(fn)
            )
        else:
            self.register = registry.register

        if instrument.PROFILE:
            COUNTERS = instrument.COUNTERS

            def profiled_dispatch(*args):
                pass

            self.dispatch = profiled_dispatch
        else:
            self.dispatch = registry.dispatch

    def interpret(self, cls, *args):
        return self.dispatch(cls, *args)(*args)


class PrioritizedInterpretation(Interpretation):

    def __init__(self, *subinterpretations):
        subinterpretations = tuple(
            ss for s in subinterpretations for ss in s.subinterpretations
        )
        assert subinterpretations
        assert len(subinterpretations) < 10, "suspicious interpretation overflow"
        assert not any(s.is_total for s in subinterpretations[:-1])
        super().__init__("/".join(s.__name__ for s in subinterpretations))
        self._subinterpretations = subinterpretations

    @property
    def subinterpretations(self):
        pass

    @property
    def is_total(self):
        pass

    @property
    def register(self):
        return self._subinterpretations[0].register

    @property
    def dispatch(self):
        return self._subinterpretations[0].dispatch

    def interpret(self, cls, *args):
        for s in self._subinterpretations:
            result = s.interpret(cls, *args)
            if result is not None:
                return result


class StatefulInterpretationMeta(type(ABC)):
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        cls.registry = KeyedRegistry(default=lambda *args: None)

        if instrument.PROFILE:
            COUNTERS = instrument.COUNTERS

            def profiled_dispatch(*args):
                pass

            cls.dispatch = staticmethod(profiled_dispatch)
        else:
            cls.dispatch = staticmethod(cls.registry.dispatch)


class StatefulInterpretation(Interpretation, metaclass=StatefulInterpretationMeta):

    def __init__(self, name="stateful"):
        super().__init__(name)

    def interpret(self, cls, *args):
        return self.dispatch(cls, *args)(self, *args)

    if instrument.DEBUG:

        @classmethod
        def register(cls, *args):
            return lambda fn: cls.registry.register(*args)(instrument.debug_logged(fn))

    else:

        @classmethod
        def register(cls, *args):
            return cls.registry.register(*args)


class Memoize(Interpretation):

    def __init__(self, base_interpretation, cache=None):
        super().__init__(f"Memoize({base_interpretation.__name__})")
        self.base_interpretation = base_interpretation
        if cache is None:
            cache = {}
        else:
            assert isinstance(cache, dict)
        self.cache = cache

    @property
    def is_total(self):
        pass

    def interpret(self, cls, *args):
        key = self.make_hash_key(cls, *args)
        value = self.cache.get(key)
        if value is None:
            self.cache[key] = value = self.base_interpretation.interpret(cls, *args)
        return value


@contextmanager
def memoize(cache=None):
    """
    Context manager wrapping :class:`Memoize` and yielding the ``cache`` dict.
    """
    base_interpretation = get_interpretation()
    with Memoize(base_interpretation, cache) as interp:
        yield interp.cache




@CallableInterpretation
def reflect(cls, *args):
    raise ValueError("Should be overwritten in terms.py")


reflect.is_total = True

normalize_base = DispatchedInterpretation("normalize")
normalize = PrioritizedInterpretation(normalize_base, reflect)
"""
Normalize modulo associativity and commutativity, but do not evaluate any
numerical operations.
"""

lazy_base = DispatchedInterpretation("lazy")
lazy = PrioritizedInterpretation(lazy_base, reflect)
"""
Performs substitutions eagerly, but construct lazy funsors for everything else.
"""

eager_base = DispatchedInterpretation("eager")
eager = PrioritizedInterpretation(eager_base, normalize_base, reflect)
"""
Eager exact naive interpretation wherever possible.
"""

die = DispatchedInterpretation("die")
eager_or_die = PrioritizedInterpretation(eager_base, die, reflect)

sequential_base = DispatchedInterpretation("sequential")
sequential = PrioritizedInterpretation(
    sequential_base, eager_base, normalize_base, reflect
)
"""
Eagerly execute ops with known implementations; additonally execute
vectorized ops sequentially if no known vectorized implementation exists.
"""

moment_matching_base = DispatchedInterpretation("moment_matching")
moment_matching = PrioritizedInterpretation(
    moment_matching_base, eager_base, normalize_base, reflect
)
"""
A moment matching interpretation of :class:`Reduce` expressions. This falls
back to :class:`eager` in other cases.
"""

push_interpretation(reflect)  # Set for optional type checking.
push_interpretation(eager)  # Use eager interpretation by default.

compress_gaussians_base = DispatchedInterpretation("compress_gaussians")
compress_gaussians = PrioritizedInterpretation(
    compress_gaussians_base, eager_base, normalize_base, reflect
)


__all__ = [
    "CallableInterpretation",
    "DispatchedInterpretation",
    "Interpretation",
    "Memoize",
    "StatefulInterpretation",
    "compress_gaussians",
    "die",
    "eager",
    "lazy",
    "memoize",
    "moment_matching",
    "normalize",
    "reflect",
    "sequential",
]
