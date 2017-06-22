import sys
import numpy

import _numpyinterface
from _numpyinterface import iterate    # this is a high-level function, should be exposed at this level

def _isstr(x):
    if sys.version_info[0] <= 2:
        return isinstance(x, basestring)
    else:
        return isinstance(x, (bytes, str))

def _xrange(x):
    if sys.version_info[0] <= 2:
        return xrange(x)
    else:
        return range(x)

def arraydict(*args, **kwds):
    if len(args) >= 2 and _isstr(args[0]) and _isstr(args[1]):
        preargs = args[0:2]
        first = 2
    else:
        preargs = ()
        first = 0

    if len(args) - first < 1:
        raise TypeError("at least one branch is required (if the first arguments are strings, they are interpreted as TFile and TTree paths)")
    
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
    dts = _numpyinterface.dtypeshape(*args, swap_bytes=swap_bytes)

    out = {}
    branchargs = {}
    for i, (name, dtype, shape) in enumerate(dts):
        out[name] = allocate(shape, dtype)
        branchargs[name] = args[first + i]

    # fill each one separately to avoid memcpys from cluster-alignment
    for name, array in out.items():
        itr = _numpyinterface.iterate(*(preargs + (branchargs[name],)), return_new_buffers=False, swap_bytes=swap_bytes)

        current = 0
        for start, end, subarray in itr:
            length = subarray.shape[0]
            array[current : current + length] = subarray
            current += length

        # now that we know how big the array is, trim it and return a view
        out[name] = trim(array, current)

    return out

def recarray(*args, **kwds):
    if len(args) >= 2 and _isstr(args[0]) and _isstr(args[1]):
        preargs = args[0:2]
        first = 2
    else:
        preargs = ()
        first = 0

    if len(args) - first < 1:
        raise TypeError("at least one branch is required (if the first arguments are strings, they are interpreted as TFile and TTree paths)")

    swap_bytes = True
    if "swap_bytes" in kwds:
        swap_bytes = kwds["swap_bytes"]
        del kwds["swap_bytes"]
    if len(kwds) > 0:
        raise TypeError("unrecognized options: " + " ".join(kwds))

    # allocate all the arrays, using a slight overestimate in their lengths (due to headers in GetTotalSize)
    dts = _numpyinterface.dtypeshape(*args, swap_bytes=swap_bytes)

    shape = None
    for n, d, s in dts:
        if len(s) != 1:
            raise ValueError("branches for a recarray must be one-dimensional")
        if shape is None:
            shape = s
        elif s[0] > shape[0]:
            shape = s

    out = numpy.empty(shape, dtype=[(bytes(n), d) for n, d, s in dts])

    branchargs = {}
    for i, (name, dtype, shape) in enumerate(dts):
        branchargs[name] = args[first + i]

    # fill each one separately to avoid memcpys from cluster-alignment
    fulllength = 0
    for name, arg in branchargs.items():
        itr = _numpyinterface.iterate(*(preargs + (arg,)), return_new_buffers=False, swap_bytes=swap_bytes)

        current = 0
        for start, end, subarray in itr:
            length = subarray.shape[0]
            out[name][current : current + length] = subarray
            current += length

        if current > fulllength:
            fulllength = current

    # now that we know how big the array is, trim it and return a view
    return out[:][:fulllength]

def iterate_pandas(*args):
    import pandas as pd

    dts = _numpyinterface.dtypeshape(*args, swap_bytes=True)
    recdtype = [(bytes(n), d) for n, d, s in dts]

    itr = _numpyinterface.iterate(*args, return_new_buffers=False, swap_bytes=True)

    for cluster in itr:
        start, end = cluster[:2]
        rec = numpy.column_stack(cluster[2:]).ravel().view(recdtype)
        yield start, end, pd.DataFrame(rec)

def pandas(*args):
    import pandas as pd

    dts = _numpyinterface.dtypeshape(*args, swap_bytes=True)
    out = pd.DataFrame(columns=[n for n, d, s in dts])

    for start, end, df in iterate_pandas(*args):
        out = out.append(df, ignore_index=True)

    return out

import time

arraydict("../../data/TrackResonanceNtuple_uncompressed.root", "twoMuon", "mass_mumu", "px")

startTime = time.time()
d = arraydict("../../data/TrackResonanceNtuple_uncompressed.root", "twoMuon", "mass_mumu", "px")
print time.time() - startTime

startTime = time.time()
r = recarray("../../data/TrackResonanceNtuple_uncompressed.root", "twoMuon", "mass_mumu", "px")
print time.time() - startTime

startTime = time.time()
p = pandas("../../data/TrackResonanceNtuple_uncompressed.root", "twoMuon", "mass_mumu", "px")
print time.time() - startTime

print numpy.array_equal(r["mass_mumu"], d["mass_mumu"])
print numpy.array_equal(r["px"], d["px"])

print numpy.array_equal(p["mass_mumu"], d["mass_mumu"])
print numpy.array_equal(p["px"], d["px"])
