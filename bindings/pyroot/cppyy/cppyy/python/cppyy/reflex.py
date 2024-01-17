""" cppyy reflection
"""

import cppyy

try:
    import __pypy__
    __all__ = [

    ]

except ImportError:
    __all__ = [
       'RETURN_TYPE',
    ]

    cppyy.include("CPyCppyy/Reflex.h")

    IS_NAMESPACE   = cppyy.gbl.Cppyy.Reflex.IS_NAMESPACE

    OFFSET         = cppyy.gbl.Cppyy.Reflex.OFFSET
    RETURN_TYPE    = cppyy.gbl.Cppyy.Reflex.RETURN_TYPE
    TYPE           = cppyy.gbl.Cppyy.Reflex.TYPE

    OPTIMAL        = cppyy.gbl.Cppyy.Reflex.OPTIMAL
    AS_TYPE        = cppyy.gbl.Cppyy.Reflex.AS_TYPE
    AS_STRING      = cppyy.gbl.Cppyy.Reflex.AS_STRING
