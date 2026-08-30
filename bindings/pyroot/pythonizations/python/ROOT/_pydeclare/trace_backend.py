# Author: Jonas Rembser CERN 08/2026

################################################################################
# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

"""The tracing backend.

Instead of reading the Python source, this backend *runs* the callable once
with symbolic arguments.  Every operator, method and numpy ufunc applied to a
symbol records itself, and what comes back is a DAG describing the computation,
which is then emitted as C++.

The trade-off against the AST backend is control flow.  A traced symbol has no
value, so `if x > 0:` cannot pick a branch; `__bool__` raises instead of
guessing.  Python-level control flow that does *not* depend on the arguments is
a different matter -- it unrolls, which is occasionally exactly what you want --
but since it silently changes what the generated code does relative to reading
the source, it has to be asked for explicitly with ``unroll=True``.
"""

import ast
import inspect
import sys
import textwrap

from . import emit
from .cpptypes import UNKNOWN_T
from .errors import PyDeclareError

_CONTROL_FLOW_NODES = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.IfExp)


class _TraceError(PyDeclareError):
    pass


def _here(func):
    """Format the position inside the traced callable, for error messages."""
    import traceback

    code = getattr(func, "__code__", None)
    frames = traceback.extract_stack()
    for frame in reversed(frames):
        if code is not None and frame.filename == code.co_filename and frame.name == code.co_name:
            return '\n  File "{}", line {}\n    {}\n'.format(frame.filename, frame.lineno, (frame.line or "").strip())
    return ""


class Sym:
    """A symbolic value standing in for one C++ expression.

    Nodes are built lazily: each carries a builder that turns the *reference
    codes* of its children into a Value.  That lets the emitter decide after
    the fact whether a shared sub-expression should become a named temporary
    rather than being pasted in twice.
    """

    __slots__ = ("ctx", "builder", "children", "value", "name", "uses")

    __array_priority__ = 1000.0

    def __init__(self, ctx, builder, children, value):
        self.ctx = ctx
        self.builder = builder
        self.children = children
        self.value = value  # eagerly built, for its type
        self.name = None  # set if this node is materialised as a temporary
        self.uses = 0
        ctx.nodes.append(self)

    # -- construction helpers ----------------------------------------------

    @property
    def type(self):
        return self.value.type

    def rebuild(self, child_values):
        return self.builder(child_values)

    def __repr__(self):
        return "Sym({!r}, {})".format(self.value.code, self.value.type)

    # -- the loud failures --------------------------------------------------

    def _refuse(self, what, hint):
        raise _TraceError(
            "{} while tracing '{}'.{}\n  {}".format(
                what, getattr(self.ctx.func, "__name__", "<lambda>"), _here(self.ctx.func), hint
            )
        )

    def __bool__(self):
        self._refuse(
            "A traced value was converted to a Python bool",
            "This happens for 'if x', 'while x', 'not x', 'x and y', 'x or y' and 'assert x'. "
            "The interpreter resolves all of them by asking the value for its truth, and a "
            "symbol has none to give: the tracing backend records what one Python run does, so "
            "it never sees the operation at all.\n"
            "  Use backend='ast', which reads the source and translates these directly. For "
            "element-wise logic the numpy spelling also works in both backends: '~x' for not, "
            "'&' for and, '|' for or, and np.where(cond, a, b) for a conditional.",
        )

    def __len__(self):
        self._refuse(
            "len() was called on a traced array",
            "The length is only known at run time. Use x.size, which stays symbolic.",
        )

    def __iter__(self):
        self._refuse(
            "A traced array was iterated over",
            "The number of elements is only known at run time. Use element-wise array "
            "expressions, or backend='ast' with an explicit loop.",
        )

    def __index__(self):
        self._refuse("A traced value was used as an index into a Python object", "Index the array itself instead.")

    __int__ = __index__

    def __float__(self):
        self._refuse(
            "A traced value was converted to a Python float",
            "Arithmetic stays symbolic; use np.* functions rather than math.* ones, which force "
            "a conversion to a Python number.",
        )

    def __hash__(self):
        return id(self)

    # -- operators ----------------------------------------------------------

    def _binop(self, op, other, swap=False):
        a, b = (self.ctx.wrap(other), self) if swap else (self, self.ctx.wrap(other))
        return self.ctx.node(lambda vals: emit.binop(op, vals[0], vals[1], self.ctx.fail), [a, b])

    def _cmp(self, op, other):
        a, b = self, self.ctx.wrap(other)
        return self.ctx.node(lambda vals: emit.compare(op, vals[0], vals[1], self.ctx.fail), [a, b])


def _install_operators():
    binary = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "truediv": "/",
        "floordiv": "//",
        "mod": "%",
        "pow": "**",
        "lshift": "<<",
        "rshift": ">>",
        "and": "&",
        "or": "|",
        "xor": "^",
    }
    for pyname, op in binary.items():

        def make(op=op, swap=False):
            def method(self, other):
                return self._binop(op, other, swap)

            return method

        setattr(Sym, "__{}__".format(pyname), make(op, False))
        setattr(Sym, "__r{}__".format(pyname), make(op, True))

    for pyname, op in {"lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "==", "ne": "!="}.items():

        def make_cmp(op=op):
            def method(self, other):
                return self._cmp(op, other)

            return method

        setattr(Sym, "__{}__".format(pyname), make_cmp(op))

    for pyname, op in {"neg": "-", "pos": "+", "invert": "~"}.items():

        def make_un(op=op):
            def method(self):
                return self.ctx.node(lambda vals: emit.unaryop(op, vals[0], self.ctx.fail), [self])

            return method

        setattr(Sym, "__{}__".format(pyname), make_un(op))

    def _abs(self):
        return self.ctx.node(lambda vals: emit.unary_function("Abs", vals[0], self.ctx.fail), [self])

    Sym.__abs__ = _abs


_install_operators()


# --- indexing ---------------------------------------------------------------


def _getitem(self, key):
    ctx = self.ctx
    if isinstance(key, slice):
        parts = [key.start, key.stop, key.step]
        syms = [None if p is None else ctx.wrap(p) for p in parts]
        children = [self] + [s for s in syms if s is not None]

        def build(vals):
            it = iter(vals[1:])
            start, stop, step = (next(it) if s is not None else None for s in syms)
            return emit.slice_(vals[0], start, stop, step, ctx.fail)

        return ctx.node(build, children)
    index = ctx.wrap(key)
    return ctx.node(lambda vals: emit.subscript(vals[0], vals[1], ctx.fail), [self, index])


Sym.__getitem__ = _getitem


def _setitem(self, key, value):
    self._refuse(
        "A traced array was assigned into",
        "The tracing backend builds a pure expression; it has nowhere to put a mutation. "
        "Use backend='ast' if you need to write into an array.",
    )


Sym.__setitem__ = _setitem


# --- attributes -------------------------------------------------------------


class _Method:
    """A method looked up on a symbol but not yet called."""

    def __init__(self, sym, name):
        self.sym = sym
        self.name = name

    def __call__(self, *args, **kwargs):
        ctx = self.sym.ctx
        if kwargs:
            raise _TraceError(
                "Keyword arguments are not supported in a call to '{}'.{}".format(self.name, _here(ctx.func))
            )
        if self.name == "astype":
            if len(args) != 1:
                raise _TraceError("astype() takes exactly one argument.{}".format(_here(ctx.func)))
            dtype = args[0]
            return ctx.node(lambda vals: emit.astype(vals[0], dtype, ctx.fail), [self.sym])
        if self.name in emit.ARRAY_METHODS and self.sym.type.is_container:
            if args:
                raise _TraceError("'{}' does not take arguments here.{}".format(self.name, _here(ctx.func)))
            return ctx.node(lambda vals: emit.array_method(vals[0], self.name, [], ctx.fail), [self.sym])
        # Anything else is passed straight through to C++: this is how methods
        # on ROOT classes such as PtEtaPhiMVector::M() are translated.
        children = [self.sym] + [ctx.wrap(a) for a in args]
        return ctx.node(lambda vals: emit.cpp_method(vals[0], self.name, vals[1:]), children)


def _getattr(self, name):
    if name.startswith("__") or name in Sym.__slots__:
        raise AttributeError(name)
    ctx = self.ctx
    if self.type.is_container:
        if name == "size":
            return ctx.node(lambda vals: emit.length(vals[0], ctx.fail), [self])
        if name in emit.ARRAY_METHODS:
            return _Method(self, name)
        if name in ("T", "shape", "ndim", "dtype", "flat", "real", "imag"):
            raise _TraceError(
                "Arrays have no '{}' in the supported subset; they are always one dimensional "
                "arrays of numbers.{}".format(name, _here(ctx.func))
            )
        return _Method(self, name)
    if self.type.is_fund:
        raise _TraceError("'{}' has no attribute '{}'.{}".format(self.type, name, _here(ctx.func)))
    # Opaque C++ object: could be a data member or a method.
    return _MaybeMember(self, name)


class _MaybeMember(emit.Value):
    """A member access on an opaque C++ object: a value, or a method to call."""

    __slots__ = ("sym", "name", "_node")

    def __init__(self, sym, name):
        ctx = sym.ctx
        node = ctx.node(lambda vals: emit.cpp_member(vals[0], name), [sym])
        super().__init__(node.value.code, node.value.type, node.value.prec)
        self.sym = sym
        self.name = name
        self._node = node

    def __call__(self, *args, **kwargs):
        # It was a method after all; drop the member node and build a call.
        ctx = self.sym.ctx
        ctx.discard(self._node)
        return _Method(self.sym, self.name)(*args, **kwargs)


Sym.__getattr__ = _getattr


# --- numpy interoperability -------------------------------------------------

_UFUNC_BINOPS = {
    "add": "+",
    "subtract": "-",
    "multiply": "*",
    "true_divide": "/",
    "divide": "/",
    "floor_divide": "//",
    "remainder": "%",
    "mod": "%",
    "power": "**",
    "float_power": "**",
    "left_shift": "<<",
    "right_shift": ">>",
    "bitwise_and": "&",
    "bitwise_or": "|",
    "bitwise_xor": "^",
    "logical_and": "&",
    "logical_or": "|",
    "logical_xor": "^",
}
_UFUNC_COMPARES = {
    "less": "<",
    "less_equal": "<=",
    "greater": ">",
    "greater_equal": ">=",
    "equal": "==",
    "not_equal": "!=",
}
_UFUNC_UNARY = {"negative": "-", "positive": "+", "invert": "~", "logical_not": "not"}


def _array_ufunc(self, ufunc, method, *inputs, **kwargs):
    ctx = self.ctx
    name = getattr(ufunc, "__name__", "")
    if method != "__call__" or kwargs.get("out") is not None:
        raise _TraceError("numpy's '{}' method of ufunc '{}' is not supported.{}".format(method, name, _here(ctx.func)))
    args = [ctx.wrap(a) for a in inputs]

    if name in _UFUNC_UNARY and len(args) == 1:
        op = _UFUNC_UNARY[name]
        return ctx.node(lambda vals: emit.unaryop(op, vals[0], ctx.fail), args)
    if name in emit.UNARY_FUNCTIONS and len(args) == 1:
        fn = emit.UNARY_FUNCTIONS[name]
        return ctx.node(lambda vals: emit.unary_function(fn, vals[0], ctx.fail), args)
    if name in _UFUNC_BINOPS and len(args) == 2:
        op = _UFUNC_BINOPS[name]
        return ctx.node(lambda vals: emit.binop(op, vals[0], vals[1], ctx.fail), args)
    if name in _UFUNC_COMPARES and len(args) == 2:
        op = _UFUNC_COMPARES[name]
        return ctx.node(lambda vals: emit.compare(op, vals[0], vals[1], ctx.fail), args)
    if name in emit.BINARY_FUNCTIONS and len(args) == 2:
        fn = emit.BINARY_FUNCTIONS[name]
        return ctx.node(lambda vals: emit.binary_function(fn, vals[0], vals[1], ctx.fail), args)
    raise _TraceError(
        "The numpy function '{}' is not available in the supported subset.{}".format(name, _here(ctx.func))
    )


def _array_function(self, func, types, args, kwargs):
    ctx = self.ctx
    name = getattr(func, "__name__", "")
    if kwargs:
        raise _TraceError("Keyword arguments to numpy's '{}' are not supported.{}".format(name, _here(ctx.func)))
    wrapped = [ctx.wrap(a) for a in args]
    if name == "where" and len(wrapped) == 3:
        return ctx.node(lambda vals: emit.ternary(vals[0], vals[1], vals[2], ctx.fail), wrapped)
    if name in emit.FREE_REDUCTIONS and len(wrapped) == 1:
        method = _REDUCTION_METHOD[emit.FREE_REDUCTIONS[name]]
        return ctx.node(lambda vals: emit.array_method(vals[0], method, [], ctx.fail), wrapped)
    raise _TraceError(
        "The numpy function '{}' is not available in the supported subset.{}".format(name, _here(ctx.func))
    )


_REDUCTION_METHOD = {fn: name for name, (fn, _) in emit.ARRAY_METHODS.items()}

Sym.__array_ufunc__ = _array_ufunc
Sym.__array_function__ = _array_function


# --- tracer-aware builtins --------------------------------------------------
#
# float(x), len(x) and friends are resolved by the interpreter to a call that
# must return a real Python object: float() insists on a float, len() on an int.
# A symbol cannot satisfy them.  They can, however, be shadowed: the traced
# callable is re-created with a __builtins__ mapping in which these names point
# at tracer-aware versions, so the ordinary Python spelling keeps working.
#
# The operations the interpreter routes through __bool__ -- 'not x', 'x and y',
# 'if x' -- have no such hook, and are reported instead.


def _traced_builtin(name):
    import builtins

    original = getattr(builtins, name)

    def wrapper(*args):
        syms = [a for a in args if isinstance(a, Sym)]
        if not syms:
            return original(*args)
        ctx = syms[0].ctx
        wrapped = [ctx.wrap(a) for a in args]

        def build(vals):
            if name in ("float", "int", "bool"):
                return emit.cast(name, vals[0], ctx.fail)
            if name == "len":
                return emit.length(vals[0], ctx.fail)
            if name == "abs":
                return emit.unary_function("Abs", vals[0], ctx.fail)
            if name == "round":
                return emit.unary_function("Round", vals[0], ctx.fail)
            if name == "sum":
                return emit.array_method(vals[0], "sum", [], ctx.fail)
            if name == "pow":
                return emit.binop("**", vals[0], vals[1], ctx.fail)
            if name in ("min", "max"):
                if len(vals) == 1:
                    return emit.array_method(vals[0], name, [], ctx.fail)
                return emit.binary_function("Maximum" if name == "max" else "Minimum", vals[0], vals[1], ctx.fail)
            ctx.fail("The builtin '{}' is not available in the supported subset".format(name))

        if name in ("float", "int", "bool", "len", "abs", "round", "sum") and len(wrapped) != 1:
            ctx.fail("{}() takes exactly one argument here".format(name))
        if name == "pow" and len(wrapped) != 2:
            ctx.fail("pow() takes exactly two arguments here")
        if name in ("min", "max") and len(wrapped) not in (1, 2):
            ctx.fail("{}() takes one or two arguments here".format(name))
        return ctx.node(build, wrapped)

    wrapper.__name__ = name
    return wrapper


class _NumpyProxy:
    """Stands in for the numpy module while tracing.

    Only np.array needs intercepting: unlike the ufuncs and the functions
    covered by __array_function__, it has no dispatch protocol, so a list
    containing symbols would silently become an object array.
    """

    def __init__(self, module):
        object.__setattr__(self, "_module", module)

    def __getattr__(self, name):
        if name == "array":
            return _traced_np_array
        return getattr(object.__getattribute__(self, "_module"), name)


def _traced_np_array(obj, dtype=None):
    import numpy as np

    values = list(obj) if isinstance(obj, (list, tuple)) else []
    syms = [v for v in values if isinstance(v, Sym)]
    if not syms:
        return np.array(obj) if dtype is None else np.array(obj, dtype=dtype)
    ctx = syms[0].ctx
    children = [ctx.wrap(v) for v in values]

    def build(vals):
        built = emit.array_from_elements(vals, ctx.fail)
        return built if dtype is None else emit.astype(built, dtype, ctx.fail)

    return ctx.node(build, children)


class _CppProxy:
    """Stands in for a C++ namespace, class or function while tracing.

    Calling it with a symbol records a C++ call; calling it with ordinary
    Python values does the real thing, so a traced callable that happens to use
    cppyy for something unrelated keeps working.
    """

    def __init__(self, cpp_name, obj):
        object.__setattr__(self, "_cpp_name", cpp_name)
        object.__setattr__(self, "_obj", obj)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        obj = object.__getattribute__(self, "_obj")
        prefix = object.__getattribute__(self, "_cpp_name")
        try:
            child = getattr(obj, name)
        except AttributeError:
            raise _TraceError("'{}' has no member '{}' known to cling".format(prefix or "the global namespace", name))
        if emit.cpp_entity_kind(child) is None:
            raise _TraceError(
                "'{}{}' is a C++ object, not a type or a function.\n"
                "  Only classes, namespaces and functions can be named from translated code. "
                "Pass the object in as an argument instead.".format(prefix + "::" if prefix else "", name)
            )
        qualified = emit.cpp_entity_name(child) or ("{}::{}".format(prefix, name) if prefix else name)
        return _CppProxy(qualified, child)

    def __call__(self, *args, **kwargs):
        obj = object.__getattribute__(self, "_obj")
        syms = [a for a in args if isinstance(a, Sym)]
        if not syms:
            return obj(*args, **kwargs)
        if kwargs:
            raise _TraceError("Keyword arguments are not supported in a call to C++")
        ctx = syms[0].ctx
        name = object.__getattribute__(self, "_cpp_name")
        children = [ctx.wrap(a) for a in args]
        return ctx.node(lambda vals: emit.call(name, vals, UNKNOWN_T), children)


TRACED_BUILTINS = ("float", "int", "bool", "len", "abs", "round", "sum", "pow", "min", "max")


def _with_traced_builtins(func):
    """A copy of func whose builtin lookups see the tracer-aware versions."""
    import builtins
    import types

    if not hasattr(func, "__code__"):
        return func
    env = dict(vars(builtins))
    for name in TRACED_BUILTINS:
        env[name] = _traced_builtin(name)
    glob = dict(getattr(func, "__globals__", {}) or {})
    glob["__builtins__"] = env
    numpy_module = sys.modules.get("numpy")
    for name, value in list(glob.items()):
        if numpy_module is not None and value is numpy_module:
            glob[name] = _NumpyProxy(numpy_module)
        elif emit.is_cpp_root_namespace(value):
            glob[name] = _CppProxy("", value)
    clone = types.FunctionType(func.__code__, glob, func.__name__, func.__defaults__, func.__closure__)
    clone.__kwdefaults__ = getattr(func, "__kwdefaults__", None)
    return clone


# --- the tracing context ----------------------------------------------------


class TraceContext:
    def __init__(self, func):
        self.func = func
        self.nodes = []

    def fail(self, msg, hint=None):
        text = "{}{}".format(msg, _here(self.func))
        if hint:
            text += "\n  " + hint
        raise _TraceError(text)

    def node(self, builder, children):
        value = builder([c.value for c in children])
        return Sym(self, builder, children, value)

    def discard(self, sym):
        # Identity, not equality: '==' on a Sym builds a comparison node.
        for i, node in enumerate(self.nodes):
            if node is sym:
                del self.nodes[i]
                return

    def leaf(self, value):
        return Sym(self, lambda vals: value, [], value)

    def wrap(self, obj):
        """Turn a Python value into a Sym, folding constants."""
        if isinstance(obj, Sym):
            return obj
        if isinstance(obj, emit.Value):
            return self.leaf(obj)
        value = emit.constant_value(obj, self.fail)
        if value is None:
            self.fail(
                "Cannot use a value of type '{}' in a traced expression".format(type(obj).__name__),
                "Only numbers, numpy scalars and one dimensional numpy arrays of numbers can be "
                "combined with traced values.",
            )
        return self.leaf(value)


def _check_no_control_flow(func, unroll):
    """Refuse to trace a function whose source contains control flow."""
    if unroll:
        return
    try:
        src = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(src)
    except (OSError, TypeError, SyntaxError):
        return  # no source: rely on Sym.__bool__ to catch the data-dependent cases
    for node in ast.walk(tree):
        if isinstance(node, _CONTROL_FLOW_NODES):
            raise _TraceError(
                "The callable contains a '{}' statement, which the tracing backend cannot "
                "translate.\n"
                '  File "{}", line {}\n'
                "  Tracing records the operations that one Python run performs, so a branch is "
                "either invisible (it depended on the arguments) or silently baked in (it did "
                "not).\n"
                "  Use backend='ast' to translate the control flow, or pass unroll=True to "
                "accept the Python-level evaluation.".format(
                    type(node).__name__.lower(),
                    getattr(func, "__code__", None) and func.__code__.co_filename,
                    node.lineno + (func.__code__.co_firstlineno - 1 if hasattr(func, "__code__") else 0),
                )
            )


# --- emission ---------------------------------------------------------------


def trace(func, param_names, param_types, declared_return, cpp_param_names, unroll=False):
    """Trace the callable and return (body, deduced_return_type)."""
    _check_no_control_flow(func, unroll)

    ctx = TraceContext(func)
    args = [ctx.leaf(emit.Value(cpp_param_names[i], param_types[i])) for i in range(len(param_names))]

    try:
        result = _with_traced_builtins(func)(*args)
    except PyDeclareError:
        raise
    except TypeError as exc:
        raise _TraceError(
            "Tracing '{}' failed: {}\n  This usually means an operation was applied to a traced "
            "value that the tracing backend does not implement.".format(getattr(func, "__name__", "<lambda>"), exc)
        )

    if isinstance(result, _MaybeMember):
        result = result._node
    if not isinstance(result, Sym):
        value = emit.constant_value(result, ctx.fail)
        if value is None:
            ctx.fail(
                "The traced callable returned a {} rather than a value derived from its arguments".format(
                    type(result).__name__
                )
            )
        result = ctx.leaf(value)

    lines, root_value = _emit(ctx, result)

    if declared_return is not None:
        if root_value.type == declared_return:
            lines.append("   return {};".format(root_value.code))
        else:
            lines.append("   return {}::Return<{}>({});".format(emit.PYD, declared_return.cpp(), root_value.code))
        deduced = declared_return
    else:
        lines.append("   return {};".format(root_value.code))
        deduced = None if root_value.type.is_unknown else root_value.type

    return "\n".join(lines), deduced


def _emit(ctx, root):
    """Emit the DAG, giving a name to every sub-expression used more than once."""
    # Count uses, reachable from the root only.
    order = []
    seen = set()

    def visit(node):
        if id(node) in seen:
            node.uses += 1
            return
        seen.add(id(node))
        node.uses = 1
        for child in node.children:
            visit(child)
        order.append(node)

    visit(root)

    lines = []
    values = {}
    for node in order:
        child_values = [values[id(c)] for c in node.children]
        value = node.rebuild(child_values)
        # A shared sub-expression is materialised so that it is computed once.
        if node.uses > 1 and node.children:
            name = "_pyd_t{}".format(len(lines) + 1)
            lines.append("   const auto {} = {};".format(name, value.code))
            values[id(node)] = emit.Value(name, value.type, emit.PREC_PRIMARY)
        else:
            values[id(node)] = value
    return lines, values[id(root)]
