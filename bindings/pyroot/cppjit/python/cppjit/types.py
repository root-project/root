"""C++ proxy types."""

import cppjit

bck = cppjit._backend
Instance = bck.CPPInstance

try:
    import __pypy__  # noqa: F401

    __all__ = ["Instance"]

except ImportError:
    __all__ = [
        "DataMember",
        "Instance",
        "Function",
        "MethodScope",  # noqa: F822
        "InstanceArray",
        "LowLevelView",
        "Template",
    ]

    DataMember = bck.CPPDataMember
    Function = bck.CPPOverload
    Method = bck.CPPOverload
    Scope = bck.CPPScope
    InstanceArray = bck.InstanceArray
    LowLevelView = bck.LowLevelView
    Template = bck.TemplateProxy

del bck
