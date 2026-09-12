from cppjit.interactive import *

# namespace at the global level
assert std

# cppjit functions
assert cppdef
assert include

try:
    import __pypy__  # noqa: F401

    # 'cppjit.gbl' bound to 'g'
    assert g
    assert g.std
except ImportError:
    # full lazy lookup available
    assert cling.runtime.gCling
