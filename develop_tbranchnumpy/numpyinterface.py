import sys
import numpy

import _numpyinterface
from _numpyinterface import iterate    # this is a high-level function, should be exposed at this level

def _isstr(x):
    if sys.version_info[0] <= 2:
        return isinstance(x, basestring)
    else:
        return isinstance(x, (bytes, str))

def dictofarrays(*args, **kwds):
    if len(args) >= 2 and _isstr(args[0]) and _isstr(args[1]):
        preargs = args[0:2]
        first = 2
    else:
        preargs = ()
        first = 0
    
    allocate = lambda shape, dtype: numpy.empty(shape, dtype=dtype)
    trim = lambda array, length: array[:length]
    swap_bytes = True

    if "allocate" in kwds:
        allocate = kwds["allocate"]
        del kwds["allocate"]
    if "trim" in kwds:
        trim = kwds["trim"]
        del kwds["trim"]
    if "swap_bytes" in kwds:
        swap_bytes = kwds["swap_bytes"]
        del kwds["swap_bytes"]
    if len(kwds) > 0:
        raise TypeError("unrecognized options: " + " ".join(kwds))

    # allocate all the arrays, using a slight overestimate in their lengths (due to headers in GetTotalSize)
    dts = _numpyinterface.dtypeshape(*args,
                                     swap_bytes=swap_bytes)
    out = {}
    branchargs = {}
    for i, (name, dtype, shape) in enumerate(dts):
        out[name] = allocate(shape, dtype)
        branchargs[name] = args[first + i]

    # allocate over each one separately to avoid memcpys from cluster-alignment
    for name, array in out.items():
        itr = _numpyinterface.iterate(*(preargs + (branchargs[name],)),
                                      return_new_buffers=False,
                                      require_alignment=False,
                                      swap_bytes=True)
        current = 0
        for start, end, subarray in itr:
            length = subarray.shape[0]
            array[current : current + length] = subarray
            current += length

        # now that we know how big the array is, trim it and return a view
        out[name] = trim(array, current)

    return out
