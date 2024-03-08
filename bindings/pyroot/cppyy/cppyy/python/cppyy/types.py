""" C++ proxy types.
"""

import cppyy

bck = cppyy._backend
Instance      = bck.CPPInstance

try:
    import __pypy__
    __all__ = [
        'Instance'
    ]

except ImportError:
    __all__ = [
        'DataMember',
        'Instance',
        'Function',
        'Method',
        'Scope',
        'InstanceArray',
        'LowLevelView',
        'Template'
    ]

    DataMember      = bck.CPPDataMember
    Function        = bck.CPPOverload
    Method          = bck.CPPOverload
    Scope           = bck.CPPScope
    InstanceArray   = bck.InstanceArray
    LowLevelView    = bck.LowLevelView
    Template        = bck.TemplateProxy

del bck
