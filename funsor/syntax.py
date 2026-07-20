
import ast
import functools
import inspect

from . import ops

PREFIX_OPERATORS = (
    ("~", ops.invert, ast.Invert),
    ("+", ops.pos, ast.UAdd),
    ("-", ops.neg, ast.USub),
)
INFIX_OPERATORS = (
    ("+", ops.add, ast.Add),
    ("-", ops.sub, ast.Sub),
    ("*", ops.mul, ast.Mult),
    ("/", ops.truediv, ast.Div),
    ("//", ops.floordiv, ast.FloorDiv),
    ("%", ops.mod, ast.Mod),
    ("**", ops.pow, ast.Pow),
    ("<<", ops.lshift, ast.LShift),
    (">>", ops.rshift, ast.RShift),
    ("|", ops.or_, ast.BitOr),
    ("^", ops.xor, ast.BitXor),
    ("&", ops.and_, ast.BitAnd),
    ("@", ops.matmul, ast.MatMult),
    ("==", ops.eq, ast.Eq),
    ("!=", ops.ne, ast.NotEq),
    ("<", ops.lt, ast.Lt),
    ("<=", ops.le, ast.LtE),
    (">", ops.gt, ast.Gt),
    (">=", ops.ge, ast.GtE),
)

PREFIX_TO_NODE = {k: v for k, _, v in PREFIX_OPERATORS}
INFIX_TO_NODE = {k: v for k, _, v in INFIX_OPERATORS}


class OpTransformer(ast.NodeTransformer):
    def __init__(self, infix, prefix, const):
        assert isinstance(infix, dict)
        assert isinstance(prefix, dict)
        assert isinstance(const, dict)
        self.infix = {INFIX_TO_NODE[k]: v for k, v in infix.items()}
        self.prefix = {PREFIX_TO_NODE[k]: v for k, v in prefix.items()}
        self.const = const

    def visit_Constant(self, node):
        pass

    def visit_UnaryOp(self, node):
        pass

    def visit_BinOp(self, node):
        pass

    def visit_Compare(self, node):
        pass


def rewrite_ops(infix={}, prefix={}, const={}):
    """
    Decorator to replace infix binary operators, prefix unary operators, and
    constants (nullary operators) in the decorated function's code with named
    variables.

    For example the following code::

        @rewrite_ops({"+": "sum_op", "*": "prod_op"})
        def product_rule(sum_op, prod_op, lhs, rhs, d):
            return d(lhs) * rhs + lhs * d(rhs)

    will be rewritten as::

        def product_rule(sum_op, prod_op, lhs, rhs, d):
            return sum_op(prod_op(d(lhs), rhs), prod_op(lhs, d(rhs)))

    .. warning:: This must be used as the innermost decorator.

    .. warning:: This requires Python 3.9+ and should not yet be used in
        Funsor library code.

    :param dict infix: An optional mapping from infix operator symbol to
        variable name.
    :param dict prefix: An optional mapping from prefix operator symbol to
        variable name.
    :param dict const: An optional mapping from constant literal to variable
        name.
    :returns: A decorator
    :rtype: callable
    """
    transformer = OpTransformer(infix, prefix, const)

    def decorator(fn):
        pass

    return decorator


__all__ = [
    "INFIX_OPERATORS",
    "PREFIX_OPERATORS",
    "rewrite_ops",
]
