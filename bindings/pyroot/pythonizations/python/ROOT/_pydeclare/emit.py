# Author: Jonas Rembser CERN 08/2026

################################################################################
# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

"""Lowering of Python operations to C++ expressions.

This module is shared by both backends: the AST walker and the tracer both
reduce to calls into the functions here, so the two produce identical C++ for
identical Python.  A backend contributes only the *discovery* of the operations
(walking a syntax tree versus recording overloaded operators); the semantics of
each individual operation live here, exactly once.
"""

from . import cpptypes as ct
from .cpptypes import BOOL_T, DOUBLE_T, LONG_T, UNKNOWN_T, CppType
from .errors import PyDeclareError

#: Namespace alias emitted into every generated snippet.
PYD = "PyD"
PYD_FULL = "ROOT::Internal::PyDeclare"

# C++ operator precedence; larger binds tighter.
PREC_PRIMARY = 17
PREC_UNARY = 16
PREC_MUL = 14
PREC_ADD = 13
PREC_SHIFT = 11
PREC_REL = 10
PREC_EQ = 9
PREC_BITAND = 8
PREC_BITXOR = 7
PREC_BITOR = 6
PREC_AND = 5
PREC_OR = 4
PREC_COND = 3

_BINOP_PREC = {
    "*": PREC_MUL,
    "/": PREC_MUL,
    "%": PREC_MUL,
    "+": PREC_ADD,
    "-": PREC_ADD,
    "<<": PREC_SHIFT,
    ">>": PREC_SHIFT,
    "<": PREC_REL,
    ">": PREC_REL,
    "<=": PREC_REL,
    ">=": PREC_REL,
    "==": PREC_EQ,
    "!=": PREC_EQ,
    "&": PREC_BITAND,
    "^": PREC_BITXOR,
    "|": PREC_BITOR,
    "&&": PREC_AND,
    "||": PREC_OR,
}


class Value:
    """A translated C++ expression together with its inferred semantic type."""

    __slots__ = ("code", "type", "prec")

    def __init__(self, code, type_=UNKNOWN_T, prec=PREC_PRIMARY):
        self.code = code
        self.type = type_ if type_ is not None else UNKNOWN_T
        self.prec = prec

    def paren(self, min_prec):
        """This expression's code, parenthesised if it binds looser than min_prec."""
        return "({})".format(self.code) if self.prec < min_prec else self.code

    def __repr__(self):
        return "Value({!r}, {})".format(self.code, self.type)


def call(name, args, type_=UNKNOWN_T):
    """A function call expression; always a primary, never needs parentheses."""
    return Value("{}({})".format(name, ", ".join(a.code for a in args)), type_, PREC_PRIMARY)


def _pyd(fn, args, type_=UNKNOWN_T):
    return call("{}::{}".format(PYD, fn), args, type_)


def _binary(op, a, b, type_):
    prec = _BINOP_PREC[op]
    # All the operators handled here are left-associative.
    return Value("{} {} {}".format(a.paren(prec), op, b.paren(prec + 1)), type_, prec)


# ---------------------------------------------------------------------------
# Literals
# ---------------------------------------------------------------------------


def literal(value, fail):
    """Translate a Python constant to a C++ literal."""
    if isinstance(value, bool):
        return Value("true" if value else "false", BOOL_T)
    if isinstance(value, int):
        if -(2**31) <= value < 2**31:
            return Value(repr(value), ct.INT_T)
        if -(2**63) <= value < 2**63:
            return Value("{}LL".format(value), ct.fund("long long"))
        fail("Integer literal {} does not fit in a 64 bit C++ integer".format(value))
    if isinstance(value, float):
        if value != value:
            return Value("std::numeric_limits<double>::quiet_NaN()", DOUBLE_T)
        if value in (float("inf"), float("-inf")):
            sign = "-" if value < 0 else ""
            return Value("{}std::numeric_limits<double>::infinity()".format(sign), DOUBLE_T)
        text = repr(value)
        if "e" not in text and "E" not in text and "." not in text and "inf" not in text:
            text += ".0"
        return Value(text, DOUBLE_T)
    if isinstance(value, complex):
        fail("Complex numbers are not supported")
    if value is None:
        fail("None is not supported; every branch must return a value")
    if isinstance(value, str):
        fail("String values are not supported")
    fail("Cannot translate a constant of type {}".format(type(value).__name__))


def sequence_literal(values, elem_type, fail):
    """Translate a Python/numpy sequence of numbers to an RVec literal."""
    parts = []
    for v in values:
        parts.append(literal(v, fail).code)
    ctype = CppType(ct.CONTAINER, elem=elem_type, container="rvec")
    return Value("{}{{{}}}".format(ctype.cpp(), ", ".join(parts)), ctype, PREC_PRIMARY)


# ---------------------------------------------------------------------------
# Binary operators
# ---------------------------------------------------------------------------


def _both_boolish(a, b):
    return a.type.is_bool and b.type.is_bool


def _check_numeric(op, a, b, fail):
    for v in (a, b):
        if v.type.kind == ct.OPAQUE:
            fail(
                "Operator '{}' is not supported for values of C++ type '{}'".format(op, v.type.name),
                hint="Only fundamental types and RVec/std::vector/std::array of them take part "
                "in arithmetic. Call a method on the object instead.",
            )


def binop(op, a, b, fail):
    """Lower a Python binary operator (ast.BinOp) to C++."""
    if op in ("+", "-", "*"):
        _check_numeric(op, a, b, fail)
        return _binary(op, a, b, ct.promote(a.type, b.type))

    if op == "/":
        _check_numeric(op, a, b, fail)
        res = ct.with_scalar(ct.promote(a.type, b.type), DOUBLE_T)
        if a.type.is_floating or b.type.is_floating:
            # C++ already promotes to floating point, so '/' is true division.
            return _binary("/", a, b, ct.promote(a.type, b.type))
        return _pyd("Div", [a, b], res)

    if op == "//":
        _check_numeric(op, a, b, fail)
        return _pyd("FloorDiv", [a, b], ct.promote(a.type, b.type))

    if op == "%":
        _check_numeric(op, a, b, fail)
        t = ct.promote(a.type, b.type)
        # For unsigned integers C++ '%' already agrees with Python's.
        if a.type.is_integral and b.type.is_integral and not a.type.is_signed and not b.type.is_signed:
            return _binary("%", a, b, t)
        return _pyd("Mod", [a, b], t)

    if op == "**":
        _check_numeric(op, a, b, fail)
        t = ct.promote(a.type, b.type)
        if t.is_integral:
            return _pyd("Pow", [a, b], t)
        return _pyd("Pow", [a, b], ct.with_scalar(t, DOUBLE_T))

    if op in ("|", "&", "^"):
        # Python/numpy overload these for element-wise logic on booleans.
        if _both_boolish(a, b):
            cpp = {"|": "||", "&": "&&", "^": "!="}[op]
            return _binary(cpp, a, b, ct.bool_like(ct.promote(a.type, b.type)))
        if a.type.is_bool != b.type.is_bool and not (a.type.is_unknown or b.type.is_unknown):
            fail(
                "Mixing a boolean and a non-boolean operand in '{}' is ambiguous".format(op),
                hint="'|' is a logical or between booleans but a bitwise or between integers. "
                "Make both operands booleans, or cast explicitly.",
            )
        if not (a.type.is_integral or a.type.is_unknown) or not (b.type.is_integral or b.type.is_unknown):
            fail("Operator '{}' requires integer or boolean operands".format(op))
        return _binary(op, a, b, ct.promote(a.type, b.type))

    if op in ("<<", ">>"):
        if not (a.type.is_integral or a.type.is_unknown) or not (b.type.is_integral or b.type.is_unknown):
            fail("Operator '{}' requires integer operands".format(op))
        return _binary(op, a, b, a.type)

    if op == "@":
        fail("The matrix multiplication operator '@' is not supported")

    fail("Binary operator '{}' is not supported".format(op))


def unaryop(op, a, fail):
    """Lower a Python unary operator to C++."""
    if op == "+":
        return Value("+{}".format(a.paren(PREC_UNARY)), a.type, PREC_UNARY)
    if op == "-":
        if a.type.kind == ct.OPAQUE:
            fail("Unary '-' is not supported for values of C++ type '{}'".format(a.type.name))
        t = a.type
        if t.is_bool:
            t = ct.with_scalar(t, ct.INT_T)
        return Value("-{}".format(a.paren(PREC_UNARY)), t, PREC_UNARY)
    if op == "not":
        return Value("!{}".format(truth(a, fail).paren(PREC_UNARY)), ct.bool_like(a.type), PREC_UNARY)
    if op == "~":
        if a.type.is_bool:
            if a.type.is_container:
                # numpy's '~' on a boolean array is a logical not.
                return Value("!{}".format(a.paren(PREC_UNARY)), a.type, PREC_UNARY)
            fail(
                "'~' on a boolean scalar is not supported",
                hint="In Python '~True' is -2, while on a numpy boolean array it is a logical "
                "not. Use 'not x' for scalars.",
            )
        if not (a.type.is_integral or a.type.is_unknown):
            fail("Operator '~' requires an integer operand")
        return Value("~{}".format(a.paren(PREC_UNARY)), a.type, PREC_UNARY)
    fail("Unary operator '{}' is not supported".format(op))


def truth(a, fail):
    """The C++ expression testing a value for Python truthiness."""
    if a.type.is_container:
        fail(
            "The truth value of an array is ambiguous",
            hint="Use .any() or .all() to reduce the array to a single boolean.",
        )
    if a.type.is_bool or a.type.is_unknown:
        return a
    if a.type.is_numeric:
        zero = Value("0", ct.INT_T)
        return _binary("!=", a, zero, BOOL_T)
    fail("Values of C++ type '{}' cannot be used as a condition".format(a.type))


def compare(op, a, b, fail):
    """Lower a single Python comparison to C++."""
    if op in ("is", "is not", "in", "not in"):
        fail(
            "The '{}' operator is not supported".format(op),
            hint="Only the arithmetic comparisons ==, !=, <, <=, > and >= can be translated.",
        )
    if a.type.kind == ct.OPAQUE or b.type.kind == ct.OPAQUE:
        # Let C++ resolve the operator on the class; we cannot infer the result.
        return _binary(op, a, b, UNKNOWN_T)
    return _binary(op, a, b, ct.bool_like(ct.promote(a.type, b.type)))


def boolop(op, values, fail):
    """Lower 'and'/'or'.  Only defined for scalars, as in numpy."""
    cpp = "&&" if op == "and" else "||"
    for v in values:
        if v.type.is_container:
            fail(
                "'{}' is not defined for arrays".format(op),
                hint="Use '&' and '|' for element-wise logic, as you would with numpy.",
            )
    out = truth(values[0], fail)
    for v in values[1:]:
        out = _binary(cpp, out, truth(v, fail), BOOL_T)
    return out


def ternary(cond, a, b, fail):
    """Lower a Python conditional expression."""
    if cond.type.is_container:
        return _pyd("Where", [cond, a, b], ct.with_scalar(cond.type, ct.promote(a.type, b.type).scalar()))
    t = ct.promote(a.type, b.type)
    code = "{} ? {} : {}".format(truth(cond, fail).paren(PREC_COND + 1), a.paren(PREC_COND + 1), b.paren(PREC_COND))
    return Value(code, t, PREC_COND)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def subscript(obj, index, fail):
    """Lower v[i] for an integer index or a boolean mask."""
    if obj.type.kind == ct.OPAQUE or obj.type.is_unknown:
        return Value("{}[{}]".format(obj.paren(PREC_PRIMARY), index.code), UNKNOWN_T, PREC_PRIMARY)
    if not obj.type.is_container:
        fail("Cannot index a value of type '{}'".format(obj.type))

    elem = obj.type.scalar()

    if index.type.is_bool and index.type.is_container:
        # RVec supports masking natively.
        return Value("{}[{}]".format(obj.paren(PREC_PRIMARY), index.code), ct.rvec_of(elem), PREC_PRIMARY)

    if index.type.is_container:
        fail(
            "Indexing with an array of indices is not supported",
            hint="Use ROOT::VecOps::Take through a boolean mask instead.",
        )

    if not (index.type.is_integral or index.type.is_unknown):
        fail("Array indices must be integers, got '{}'".format(index.type))

    # A non-negative literal, or an unsigned index, needs no wrap-around check.
    literal_nonneg = index.code.isdigit()
    if literal_nonneg or (index.type.is_integral and not index.type.is_signed):
        return Value("{}[{}]".format(obj.paren(PREC_PRIMARY), index.code), elem, PREC_PRIMARY)
    return _pyd("Index", [obj, index], elem)


def slice_(obj, start, stop, step, fail):
    """Lower v[a:b:c] with full Python slice semantics."""
    if not obj.type.is_container:
        if obj.type.kind == ct.OPAQUE:
            fail("Cannot slice a value of C++ type '{}'".format(obj.type.name))
        fail("Cannot slice a value of type '{}'".format(obj.type))

    elem = obj.type.scalar()
    out_t = ct.rvec_of(elem)

    # v[::-1] is a plain reverse; emit the readable form.
    if start is None and stop is None and step is not None and step.code == "-1":
        return call("ROOT::VecOps::Reverse", [obj], out_t)
    if start is None and stop is None and step is None:
        return Value(obj.code, out_t, obj.prec)

    zero = Value("0", LONG_T)
    one = Value("1", LONG_T)
    args = [
        obj,
        start if start is not None else zero,
        stop if stop is not None else zero,
        step if step is not None else one,
        Value("true" if start is not None else "false", BOOL_T),
        Value("true" if stop is not None else "false", BOOL_T),
    ]
    return _pyd("Slice", args, out_t)


# ---------------------------------------------------------------------------
# Methods and functions
# ---------------------------------------------------------------------------

#: numpy-style reductions available as methods on arrays.
ARRAY_METHODS = {
    "sum": ("Sum", "scalar"),
    "prod": ("Prod", "scalar"),
    "mean": ("Mean", "double"),
    "std": ("Std", "double"),
    "min": ("Min", "scalar"),
    "max": ("Max", "scalar"),
    "argmin": ("ArgMin", "long"),
    "argmax": ("ArgMax", "long"),
    "any": ("Any", "bool"),
    "all": ("All", "bool"),
}

#: numpy / math functions, mapped to the C++ support library.
UNARY_FUNCTIONS = {
    "abs": "Abs",
    "absolute": "Abs",
    "fabs": "Abs",
    "sqrt": "Sqrt",
    "cbrt": "Cbrt",
    "exp": "Exp",
    "exp2": "Exp2",
    "expm1": "Expm1",
    "log": "Log",
    "log2": "Log2",
    "log10": "Log10",
    "log1p": "Log1p",
    "sin": "Sin",
    "cos": "Cos",
    "tan": "Tan",
    "arcsin": "Asin",
    "asin": "Asin",
    "arccos": "Acos",
    "acos": "Acos",
    "arctan": "Atan",
    "atan": "Atan",
    "sinh": "Sinh",
    "cosh": "Cosh",
    "tanh": "Tanh",
    "arcsinh": "Asinh",
    "asinh": "Asinh",
    "arccosh": "Acosh",
    "acosh": "Acosh",
    "arctanh": "Atanh",
    "atanh": "Atanh",
    "floor": "Floor",
    "ceil": "Ceil",
    "trunc": "Trunc",
    "rint": "Round",
    "round": "Round",
    "erf": "Erf",
    "erfc": "Erfc",
    "lgamma": "Lgamma",
    "gamma": "Tgamma",
    "tgamma": "Tgamma",
}

#: Functions whose result is always a double, whatever the argument type.
_DOUBLE_RESULT = frozenset(
    [
        "Sqrt",
        "Cbrt",
        "Exp",
        "Exp2",
        "Expm1",
        "Log",
        "Log2",
        "Log10",
        "Log1p",
        "Sin",
        "Cos",
        "Tan",
        "Asin",
        "Acos",
        "Atan",
        "Sinh",
        "Cosh",
        "Tanh",
        "Asinh",
        "Acosh",
        "Atanh",
        "Erf",
        "Erfc",
        "Lgamma",
        "Tgamma",
        "Atan2",
        "Hypot",
        "Fmod",
    ]
)

BINARY_FUNCTIONS = {
    "arctan2": "Atan2",
    "atan2": "Atan2",
    "hypot": "Hypot",
    "fmod": "Fmod",
    "maximum": "Maximum",
    "minimum": "Minimum",
    "power": "Pow",
    "pow": "Pow",
}

#: numpy reductions that also exist as free functions.
FREE_REDUCTIONS = {
    "sum": "Sum",
    "prod": "Prod",
    "mean": "Mean",
    "average": "Mean",
    "std": "Std",
    "amin": "Min",
    "amax": "Max",
    "argmin": "ArgMin",
    "argmax": "ArgMax",
    "any": "Any",
    "all": "All",
}

#: numpy constants.
CONSTANTS = {
    "pi": ("M_PI", DOUBLE_T),
    "e": ("M_E", DOUBLE_T),
    "inf": ("std::numeric_limits<double>::infinity()", DOUBLE_T),
    "nan": ("std::numeric_limits<double>::quiet_NaN()", DOUBLE_T),
    "tau": ("(2. * M_PI)", DOUBLE_T),
}


def _reduction_type(kind, obj):
    if kind == "double":
        return DOUBLE_T
    if kind == "long":
        return LONG_T
    if kind == "bool":
        return BOOL_T
    elem = obj.type.scalar()
    if elem.is_bool:
        return LONG_T
    return elem


def array_method(obj, name, args, fail):
    """Lower a numpy-style method call on an array."""
    fn, kind = ARRAY_METHODS[name]
    if args:
        fail(
            "'{}' does not accept arguments here".format(name),
            hint="Reductions over an axis are not supported; the arrays are one dimensional.",
        )
    return _pyd(fn, [obj], _reduction_type(kind, obj))


def unary_function(fn, arg, fail):
    """Lower a unary numpy/math function."""
    result = ct.with_scalar(arg.type, DOUBLE_T) if fn in _DOUBLE_RESULT else arg.type
    return _pyd(fn, [arg], result)


def binary_function(fn, a, b, fail):
    """Lower a binary numpy/math function."""
    t = ct.promote(a.type, b.type)
    if fn in _DOUBLE_RESULT:
        t = ct.with_scalar(t, DOUBLE_T)
    return _pyd(fn, [a, b], t)


def cast(target, arg, fail):
    """Lower float(x) / int(x) / bool(x)."""
    if arg.type.is_container:
        fail(
            "Cannot convert an array with '{}()'".format(target),
            hint="Use .astype(...) on a numpy array, or reduce the array first.",
        )
    if target == "float":
        return Value("static_cast<double>({})".format(arg.code), DOUBLE_T, PREC_UNARY)
    if target == "int":
        # Python's int() truncates towards zero, and so does a C++ cast.
        return Value("static_cast<long>({})".format(arg.code), LONG_T, PREC_UNARY)
    if target == "bool":
        return truth(arg, fail)
    fail("Conversion to '{}' is not supported".format(target))


def length(arg, fail):
    """Lower len(x)."""
    if not (arg.type.is_container or arg.type.is_unknown or arg.type.kind == ct.OPAQUE):
        fail("len() is only defined for arrays, got '{}'".format(arg.type))
    return _pyd("Len", [arg], LONG_T)


def cpp_method(obj, name, args):
    """Pass a method call through to C++ untouched.

    This is what makes arbitrary ROOT classes usable: the transpiler does not
    need to know anything about PtEtaPhiMVector to translate 'v.M()'.
    """
    return Value(
        "{}.{}({})".format(obj.paren(PREC_PRIMARY), name, ", ".join(a.code for a in args)),
        UNKNOWN_T,
        PREC_PRIMARY,
    )


def cpp_member(obj, name):
    """Pass a data member access through to C++ untouched."""
    return Value("{}.{}".format(obj.paren(PREC_PRIMARY), name), UNKNOWN_T, PREC_PRIMARY)


def dtype_to_cpp(obj, fail):
    """Map a numpy dtype, scalar type or C++ type name to a C++ type name."""
    if isinstance(obj, str):
        parsed = ct.parse_type(obj)
        if parsed.kind == ct.FUND:
            return parsed.name
        cpp = ct.NUMPY_TO_CPP.get(obj)
        if cpp is not None:
            return cpp
        fail("'{}' is not a supported element type".format(obj))
    try:
        import numpy as np

        name = np.dtype(obj).name
    except Exception:
        fail("Cannot interpret {!r} as an element type".format(obj))
    cpp = ct.NUMPY_TO_CPP.get(name)
    if cpp is None:
        fail("The element type '{}' has no supported C++ counterpart".format(name))
    return cpp


def array_from_elements(values, fail):
    """Build an RVec out of individually computed elements, as np.array does."""
    if not values:
        fail("An empty array cannot be built: its element type would be unknown")
    elem = values[0].type
    for v in values[1:]:
        elem = ct.promote(elem, v.type)
    if elem.is_container:
        fail("Nested arrays are not supported; only one dimensional arrays can be built")
    if elem.is_unknown:
        fail(
            "Cannot determine the element type of the array",
            hint="Give the elements a known type, or pass dtype=... via .astype().",
        )
    type_ = ct.rvec_of(elem)
    return Value("{}{{{}}}".format(type_.cpp(), ", ".join(v.code for v in values)), type_, PREC_PRIMARY)


def astype(obj, dtype_obj, fail):
    """Lower x.astype(dtype)."""
    cpp = dtype_to_cpp(dtype_obj, fail)
    if not obj.type.is_container:
        if obj.type.is_unknown or obj.type.is_numeric:
            return Value("static_cast<{}>({})".format(cpp, obj.code), ct.fund(cpp), PREC_UNARY)
        fail("astype() is not defined for values of type '{}'".format(obj.type))
    target = ct.rvec_of(ct.fund(cpp))
    if obj.type == target:
        return obj
    return Value(
        "{}::Return<{}>({})".format(PYD, target.cpp(), obj.code),
        target,
        PREC_PRIMARY,
    )


def check_no_kwargs(kwargs, what, fail):
    if kwargs:
        fail("Keyword arguments are not supported in a call to {}".format(what))


def unsupported(msg, hint=None):
    raise PyDeclareError(msg if hint is None else "{}\n  {}".format(msg, hint))


def constant_value(obj, fail):
    """Convert a Python object captured from the enclosing scope to a C++ literal."""
    if isinstance(obj, bool):
        return literal(obj, fail)
    if isinstance(obj, (int, float)):
        return literal(obj, fail)

    try:
        import numpy as np
    except ImportError:
        np = None

    if np is not None:
        if isinstance(obj, np.generic):
            cpp = ct.NUMPY_TO_CPP.get(obj.dtype.name)
            if cpp is None:
                return None
            value = literal(obj.item(), fail)
            return Value("static_cast<{}>({})".format(cpp, value.code), ct.fund(cpp), PREC_UNARY)
        if isinstance(obj, np.ndarray):
            if obj.ndim != 1:
                return None
            cpp = ct.NUMPY_TO_CPP.get(obj.dtype.name)
            if cpp is None:
                return None
            return sequence_literal([x.item() for x in obj], ct.fund(cpp), fail)

    if (
        isinstance(obj, (list, tuple))
        and obj
        and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in obj)
    ):
        elem = DOUBLE_T if any(isinstance(x, float) for x in obj) else LONG_T
        return sequence_literal(list(obj), elem, fail)
    if isinstance(obj, (list, tuple)) and obj and all(isinstance(x, bool) for x in obj):
        return sequence_literal(list(obj), BOOL_T, fail)

    return None
