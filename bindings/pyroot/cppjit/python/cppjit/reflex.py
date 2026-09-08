"""cppjit reflection"""

import cppjit

try:
    import __pypy__  # noqa: F401

    __all__ = []

except ImportError:
    __all__ = [
        "RETURN_TYPE",
    ]

    cppjit.include("cpyrt/Reflex.h")

    IS_NAMESPACE = cppjit.gbl.cppjit.interop.Reflex.IS_NAMESPACE
    IS_AGGREGATE = cppjit.gbl.cppjit.interop.Reflex.IS_AGGREGATE

    OFFSET = cppjit.gbl.cppjit.interop.Reflex.OFFSET
    RETURN_TYPE = cppjit.gbl.cppjit.interop.Reflex.RETURN_TYPE
    TYPE = cppjit.gbl.cppjit.interop.Reflex.TYPE

    OPTIMAL = cppjit.gbl.cppjit.interop.Reflex.OPTIMAL
    AS_TYPE = cppjit.gbl.cppjit.interop.Reflex.AS_TYPE
    AS_STRING = cppjit.gbl.cppjit.interop.Reflex.AS_STRING
