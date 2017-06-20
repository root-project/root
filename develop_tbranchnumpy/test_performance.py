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

def numpy_momentum(reps, return_new_buffers, swap_bytes):
    iterator = numpyinterface.iterate(
        "TrackResonanceNtuple.root", "TrackResonanceNtuple/twoMuon", "px", "py", "pz",
        return_new_buffers = return_new_buffers,
        swap_bytes = swap_bytes)

    startTime = time.time()

    for i in xrange(reps):
        total = 0
        for start, end, px, py, pz in iterator:
            total = numpy.sqrt(px**2 + py**2 + pz**2).sum()

    endTime = time.time()

    return endTime - startTime

def numba_momentum(reps, return_new_buffers, swap_bytes):
    iterator = numpyinterface.iterate(
        "TrackResonanceNtuple.root", "TrackResonanceNtuple/twoMuon", "px", "py", "pz",
        return_new_buffers = return_new_buffers,
        swap_bytes = swap_bytes)

    startTime = time.time()

    for i in xrange(reps):
        total = 0
        for start, end, px, py, pz in iterator:
            total = momentumsum(px, py, pz)

    endTime = time.time()

    return endTime - startTime

print "numpy big-endian:"
numpy_momentum(5, False, False)          # warm up
print numpy_momentum(100, False, False)  # real run

print "numpy little-endian:"
numpy_momentum(5, False, True)           # warm up
print numpy_momentum(100, False, True)   # real run

print "numba little-endian:"
numba_momentum(5, False, True)           # warm up
print numba_momentum(100, False, True)   # real run

print numpyinterface.performance()
