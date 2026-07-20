
import atexit
import functools
import inspect
import os
from collections import Counter, defaultdict
from timeit import default_timer

DEBUG = int(os.environ.get("FUNSOR_DEBUG", 0))
PROFILE = int(os.environ.get("FUNSOR_PROFILE", 0))

STACK_SIZE = 0


def get_indent():
    result = "    \u2502" * (STACK_SIZE // 4 + 3)
    return result[:STACK_SIZE]


if DEBUG:

    class DebugLogged(object):
        def __init__(self, fn):
            self.fn = fn
            while isinstance(fn, functools.partial):
                fn = fn.func
            path = inspect.getabsfile(fn)
            lineno = inspect.getsourcelines(fn)[1]
            self._message = "{} file://{} {}".format(fn.__name__, path, lineno)

        def __call__(self, *args, **kwargs):
            global STACK_SIZE
            print(get_indent() + self._message)
            STACK_SIZE += 1
            try:
                return self.fn(*args, **kwargs)
            finally:
                STACK_SIZE -= 1

        @property
        def register(self):
            return self.fn.register

    def debug_logged(fn):
        if isinstance(fn, DebugLogged):
            return fn
        return DebugLogged(fn)

elif PROFILE:

    class ProfileLogged(object):
        def __init__(self, fn):
            self.fn = fn
            while isinstance(fn, functools.partial):
                fn = fn.func
            path = inspect.getabsfile(fn).split("/funsor/")[-1]
            lineno = inspect.getsourcelines(fn)[1]
            self._message = "{} {} {}".format(fn.__name__, path, lineno)

        def __call__(self, *args, **kwargs):
            start = default_timer()
            result = self.fn(*args, **kwargs)
            COUNTERS["time"][self._message] += default_timer() - start
            COUNTERS["call"][self._message] += 1
            return result

        @property
        def register(self):
            return self.fn.register

    def debug_logged(fn):
        if isinstance(fn, ProfileLogged):
            return fn
        return ProfileLogged(fn)

else:

    def debug_logged(fn):
        return fn


profile = __builtins__.get("profile", debug_logged)


COUNTERS = defaultdict(Counter)
if PROFILE:
    COUNTERS["time"]["total"] -= default_timer()

    @atexit.register
    def print_counters():
        pass


__all__ = [
    "DEBUG",
    "PROFILE",
    "STACK_SIZE",
    "debug_logged",
    "get_indent",
    "profile",
]
