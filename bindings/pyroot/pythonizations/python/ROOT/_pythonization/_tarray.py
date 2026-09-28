# Author: Enric Tejedor CERN  11/2018

################################################################################
# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

r'''
\pythondoc TArray

When used from Python, the subclasses of TArray (TArrayC, TArrayS, TArrayI, TArrayL, TArrayF and TArrayD) benefit from the following extra features:

- Their size can be obtained with `len`, which is equivalent to TArray::GetSize():
\code{.py}
import ROOT

a = ROOT.TArrayD(2)
print(len(a)) # prints '2'
\endcode

- Their elements can be read and written with the `getitem` and `setitem` operators, respectively:
\code{.py}
a[0] = 0.2
a[1] = 1.7
print(a[0]) # prints '0.2'
\endcode

- They are iterable:
\code{.py}
for elem in a:
    print(elem)
\endcode

\endpythondoc
'''

import sys

from . import pythonization
from ._generic import _add_getitem_checked

_tarray_format_map = {
    "TArrayC": ("b", 1),
    "TArrayS": ("h", 2),
    "TArrayI": ("i", 4),
    "TArrayL": ("l", 8 if sys.platform != "win32" else 4),
    "TArrayL64": ("q", 8),
    "TArrayF": ("f", 4),
    "TArrayD": ("d", 8),
}


def _get_buffer_for_tarray(self, flags=0):
    import ctypes
    import ROOT

    classname = type(self).__name__
    size = self.GetSize()
    if size == 0:
        return memoryview(bytearray(0))

    for name_prefix, (fmt, itemsize) in _tarray_format_map.items():
        if classname.startswith(name_prefix):
            addr = ROOT._cppyy.ll.addressof(self.GetArray())
            total_bytes = size * itemsize
            PyMemoryView_FromMemory = ctypes.pythonapi.PyMemoryView_FromMemory
            PyMemoryView_FromMemory.restype = ctypes.py_object
            PyMemoryView_FromMemory.argtypes = [ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_int]
            raw_view = PyMemoryView_FromMemory(addr, total_bytes, 0x0200)
            return raw_view.cast(fmt)

    raise BufferError(f"Buffer protocol not supported for type {classname}")


@pythonization("TArray", is_prefix=True)
def pythonize_tarray(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: string containing the name of the class

    if not name == 'TArray':
        # Add checked __getitem__. It has to be directly added to the TArray
        # subclasses, which have a default __getitem__.
        # The new __getitem__ allows to throw pythonic IndexError when index
        # is out of range and to iterate over the array.
        _add_getitem_checked(klass)

        # Add Python buffer protocol support (PEP 688)
        klass.to_memoryview = _get_buffer_for_tarray
        if sys.version_info >= (3, 12):
            klass.__buffer__ = _get_buffer_for_tarray
