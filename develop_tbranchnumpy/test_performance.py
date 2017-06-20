import math
import time

import numpy
import numba

import numpyinterface

@numba.jit(nopython=True)
def momentumsum(px, py, pz):
    total = 0.0
    for i in range(len(px)):
        total += math.sqrt(px[i]**2 + py[i]**2 + pz[i]**2)
    return total

def numpy_momentum(reps, fileName, return_new_buffers, swap_bytes):
    iterator = numpyinterface.iterate(
        fileName, "twoMuon", "px", "py", "pz",
        return_new_buffers = return_new_buffers,
        swap_bytes = swap_bytes)

    startTime = time.time()

    for i in xrange(reps):
        total = 0
        for start, end, px, py, pz in iterator:
            total = numpy.sqrt(px**2 + py**2 + pz**2).sum()

    endTime = time.time()

    return endTime - startTime

def numba_momentum(reps, fileName, return_new_buffers, swap_bytes):
    iterator = numpyinterface.iterate(
        fileName, "twoMuon", "px", "py", "pz",
        return_new_buffers = return_new_buffers,
        swap_bytes = swap_bytes)

    startTime = time.time()

    for i in xrange(reps):
        total = 0
        for start, end, px, py, pz in iterator:
            total = momentumsum(px, py, pz)

    endTime = time.time()

    return endTime - startTime

WARM_UP = 5
REPS = 100

for fileName in "TrackResonanceNtuple_uncompressed.root", "TrackResonanceNtuple_compressed.root":
    for label, return_new_buffers in ("new buffers", True), ("views", False):
        print fileName, "numpy big-endian", label,
        numpy_momentum(WARM_UP, fileName, return_new_buffers, False)     # warm up
        print numpy_momentum(REPS, fileName, return_new_buffers, False)  # real run

        print fileName, "numpy little-endian", label,
        numpy_momentum(WARM_UP, fileName, return_new_buffers, True)      # warm up
        print numpy_momentum(REPS, fileName, return_new_buffers, True)   # real run

        print fileName, "numba little-endian", label,
        numba_momentum(WARM_UP, fileName, return_new_buffers, True)      # warm up
        print numba_momentum(REPS, fileName, return_new_buffers, True)   # real run

print
print numpyinterface.performance()
