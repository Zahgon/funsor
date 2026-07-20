
from funsor.util import get_backend, set_backend


class OpProgram:

    def __init__(self, constants, inputs, operations):
        super().__init__()
        self.constants = tuple(constants)
        self.inputs = tuple(inputs)
        self.operations = tuple(operations)
        self.backend = get_backend()

    def __call__(self, **kwargs):
        set_backend(self.backend)

        env = list(self.constants)

        for name in self.inputs:
            value = kwargs.pop(name, None)
            if value is None:
                raise ValueError(f"Missing kwarg: {repr(name)}")
            env.append(value)
        if kwargs:
            raise ValueError(f"Unrecognized kwargs: {set(kwargs)}")

        for op, arg_ids in self.operations:
            args = tuple(env[i] for i in arg_ids)
            value = op(*args)
            env.append(value)

        result = env[-1]
        return result

    def as_code(self, name="program"):
        pass


def make_tuple(*args):
    pass


def _print_op(op):
    pass


__all__ = [
    "OpProgram",
]
