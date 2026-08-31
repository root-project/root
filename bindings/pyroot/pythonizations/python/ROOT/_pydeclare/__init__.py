# Author: Jonas Rembser CERN 08/2026

################################################################################
# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

"""Declare C++ functions from Python callables by translating them to C++.

This is a JIT-free alternative to ``ROOT.Numba.Declare``: instead of compiling
the Python callable with numba and calling into it through a raw function
pointer, the callable is translated to C++ source and handed to cling.  The
generated code is therefore inlinable into the RDataFrame event loop, can call
anything cling knows about (including user classes), and needs no third-party
dependency.

The translation walks the Python syntax tree.  Because the argument types are
known -- RDataFrame knows its column types, and the decorator takes them
explicitly -- forward type inference through the tree is enough to choose the
right C++ for every operation, including control flow.

Usage::

    @ROOT.Py.Declare(["float", "int"], "float")
    def pypow(x, y):
        return x**y

    ROOT.RDataFrame(4).Define("x", "(float)rdfentry_").Define("y", "Py::pypow(x, 3)")

The generated C++ is available on the callable as ``__cpp_wrapper__``.
"""

import inspect
import os

from . import cpptypes as ct
from . import emit, support
from .ast_backend import CPP_KEYWORDS, AstTranspiler
from .errors import PyDeclareError

__all__ = ["Declare", "PyDeclareError", "declare"]

DEFAULT_NAMESPACE = "Py"

#: (namespace, name) -> generated code, so that re-running a script in the same
#: session does not hit a cling redefinition error.
_DECLARED = {}


def _sanitize(name):
    if name in CPP_KEYWORDS or name.startswith("PyD"):
        return name + "_"
    return name


def _param_declaration(raw, type_, cpp_name):
    """The C++ parameter declaration for one argument.

    Fundamental types are taken by value.  Everything else is taken by const
    reference unless the user explicitly wrote a reference in the signature, in
    which case that spelling is honoured.
    """
    if type_.kind == ct.FUND:
        return "{} {}".format(type_.cpp(), cpp_name)
    raw_s = str(raw).strip()
    if raw_s.endswith("&"):
        qualifier = "const " if raw_s.startswith("const") else ""
        return "{}{} &{}".format(qualifier, type_.cpp(), cpp_name)
    return "const {} &{}".format(type_.cpp(), cpp_name)


def _signature_names(func, n_expected):
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        raise PyDeclareError("Cannot inspect the signature of {!r}".format(func))
    names = []
    for param in signature.parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise PyDeclareError("*args and **kwargs are not supported")
        if param.kind == param.KEYWORD_ONLY:
            raise PyDeclareError("Keyword-only arguments are not supported")
        names.append(param.name)
    if len(names) != n_expected:
        raise PyDeclareError(
            "The callable takes {} argument(s) but {} input type(s) were declared".format(len(names), n_expected)
        )
    return names


def declare(func, input_types, return_type=None, name=None, namespace=DEFAULT_NAMESPACE):
    """Translate a Python callable to C++ and declare it to cling.

    Returns the original callable, with the generated code and some metadata
    attached as attributes.
    """
    raw_types = list(input_types or [])
    param_names = _signature_names(func, len(raw_types))
    parsed = [ct.parse_type(t) for t in raw_types]
    declared_return = ct.parse_type(return_type) if return_type is not None else None
    if declared_return is not None and declared_return.is_unknown:
        declared_return = None

    cpp_params = [_sanitize(n) for n in param_names]
    declarations = [_param_declaration(raw_types[i], parsed[i], cpp_params[i]) for i in range(len(parsed))]

    # std::vector and std::array are viewed as RVec inside the body, so that the
    # element-wise operators are available for all supported containers.
    adapters = []
    body_types = []
    body_names = []
    for i, t in enumerate(parsed):
        if t.is_container and t.container != "rvec":
            view = cpp_params[i] + "_v"
            adapters.append("   const auto {} = {}::AsRVec({});".format(view, emit.PYD, cpp_params[i]))
            body_names.append(view)
            body_types.append(ct.rvec_of(t.scalar()))
        else:
            body_names.append(cpp_params[i])
            body_types.append(t)

    transpiler = AstTranspiler(func, param_names, body_types, declared_return, body_names)
    body, deduced = transpiler.translate()

    cpp_return = (declared_return or deduced).cpp() if (declared_return or deduced) is not None else "auto"
    func_name = name or getattr(func, "__name__", None)
    if not func_name or func_name == "<lambda>":
        raise PyDeclareError("Anonymous callables need an explicit name=... argument")

    code = _render(namespace, func_name, cpp_return, declarations, adapters, body, func)

    key = (namespace, func_name)
    previous = _DECLARED.get(key)
    if previous is None:
        support.ensure_declared()
        import ROOT

        if not ROOT.gInterpreter.Declare(code):
            raise PyDeclareError("cling rejected the generated code:\n\n{}".format(code))
        _DECLARED[key] = code
    elif previous != code:
        raise PyDeclareError(
            "'{}::{}' was already declared from a different Python callable in this session.\n"
            "  Pass name=... to give this one a different C++ name.".format(namespace, func_name)
        )

    func.__cpp_wrapper__ = code
    func.__pydeclare_cpp_name__ = "{}::{}".format(namespace, func_name)
    func.__pydeclare_return_type__ = declared_return or deduced
    return func


def _render(namespace, func_name, cpp_return, declarations, adapters, body, func):
    origin = ""
    try:
        origin = " ({}:{})".format(os.path.basename(inspect.getsourcefile(func) or "?"), func.__code__.co_firstlineno)
    except (OSError, TypeError, AttributeError):
        pass
    header = [
        "namespace {} {{".format(namespace),
        "",
        "namespace PyD = ROOT::Internal::PyDeclare;",
        "",
        "/// Generated from the Python callable '{}'{} by the ROOT".format(
            getattr(func, "__name__", func_name), origin
        ),
        "/// Python-to-C++ transpiler.",
        "{} {}({})".format(cpp_return, func_name, ", ".join(declarations)),
        "{",
    ]
    return "\n".join(header + adapters + [body, "}", "", "}} // namespace {}".format(namespace)]) + "\n"


def Declare(input_types=None, return_type=None, name=None, namespace=DEFAULT_NAMESPACE):
    """Decorator making a Python callable available in C++, by translation.

    Arguments mirror ``ROOT.Numba.Declare``:

    input_types
        List of C++ type names, one per argument of the callable.
    return_type
        The C++ return type.  If omitted, it is deduced -- either by the
        transpiler, or by the C++ compiler through an ``auto`` return type.
    name
        The name of the generated C++ function; defaults to the Python name.
    namespace
        The C++ namespace to declare it in; defaults to ``Py``.
    """

    def inner(func):
        return declare(
            func,
            input_types,
            return_type=return_type,
            name=name,
            namespace=namespace,
        )

    return inner


def _numba_declare_dispatch(input_types, return_type=None, name=None, **kwargs):
    """``ROOT.Numba.Declare``, routed to the implementation selected by
    ``$ROOT_NUMBA_DECLARE_BACKEND``: ``numba`` (the default, the original
    implementation) or ``ast``.  The generated function always lands in the
    ``Numba`` C++ namespace, so existing code that says ``"Numba::myfunc(x)"``
    keeps working with either implementation.
    """
    backend = os.environ.get("ROOT_NUMBA_DECLARE_BACKEND", "numba").strip().lower()
    if backend in ("", "numba"):
        from .._numbadeclare import _NumbaDeclareDecorator

        return _NumbaDeclareDecorator(input_types, return_type, name)
    if backend != "ast":
        raise PyDeclareError("ROOT_NUMBA_DECLARE_BACKEND is '{}'; expected 'numba' or 'ast'".format(backend))
    return Declare(input_types, return_type, name, namespace="Numba", **kwargs)
