import math
import time

import numpy
import numba

import numpyinterface

def numpy_momentum(reps, return_new_buffers, swap_bytes):
    total = 0

    startTime = time.time()
    for i in xrange(reps):
        for start, end, px, py, pz in numpyinterface.iterate(
            "TrackResonanceNtuple.root", "TrackResonanceNtuple/twoMuon", "px", "py", "pz",
            return_new_buffers = return_new_buffers,
            swap_bytes = swap_bytes):

            total += numpy.sqrt(px**2 + py**2 + pz**2).sum()

    endTime = time.time()
    return endTime - startTime

@numba.jit(nopython=True)
def update(total, px, py, pz):
    for i in range(len(px)):
        total += math.sqrt(px[i]**2 + py[i]**2 + pz[i]**2)
    return total

def numba_momentum(reps, return_new_buffers, swap_bytes):
    total = 0
    
    startTime = time.time()
    for i in xrange(reps):
        for start, end, px, py, pz in numpyinterface.iterate(
            "TrackResonanceNtuple.root", "TrackResonanceNtuple/twoMuon", "px", "py", "pz",
            return_new_buffers = return_new_buffers,
            swap_bytes = swap_bytes):

            total = update(total, px, py, pz)

    endTime = time.time()
    return endTime - startTime

print "numpy big-endian:"
numpy_momentum(3, False, False)         # warm up
print numpy_momentum(10, False, False)  # real run

print "numpy little-endian:"
numpy_momentum(3, False, True)          # warm up
print numpy_momentum(10, False, True)   # real run

print "numba little-endian:"
numba_momentum(3, False, True)          # warm up
print numba_momentum(10, False, True)   # real run

print numpyinterface.performance()
