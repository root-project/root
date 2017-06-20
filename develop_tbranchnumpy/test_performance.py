import math
import time

import numpy
import numba
import ROOT
import root_numpy

import numpyinterface

@numba.jit(nopython=True)
def momentumsum(px, py, pz):
    total = 0.0
    for i in range(len(px)):
        total += math.sqrt(px[i]**2 + py[i]**2 + pz[i]**2)
    return total

@numba.jit(nopython=True)
def energysum(px, py, pz, mass_mumu):
    total = 0.0
    for i in range(len(px)):
        total += math.sqrt(px[i]**2 + py[i]**2 + pz[i]**2 + mass_mumu[i]**2)
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

def rootnumpy_momentum(reps, fileName):
    file = ROOT.TFile(fileName)
    tree = file.Get("twoMuon")

    startTime = time.time()

    for i in xrange(reps):
        array = root_numpy.tree2array(tree, branches=["px", "py", "pz"])
        px = array["px"]
        py = array["py"]
        pz = array["pz"]
        total = numpy.sqrt(px**2 + py**2 + pz**2).sum()

    endTime = time.time()

    return endTime - startTime

def numpy_energy(reps, fileName, return_new_buffers, swap_bytes):
    iterator = numpyinterface.iterate(
        fileName, "twoMuon", "px", "py", "pz", "mass_mumu",
        return_new_buffers = return_new_buffers,
        swap_bytes = swap_bytes)

    startTime = time.time()

    for i in xrange(reps):
        total = 0
        for start, end, px, py, pz, mass_mumu in iterator:
            total = numpy.sqrt(px**2 + py**2 + pz**2 + mass_mumu**2).sum()

    endTime = time.time()

    return endTime - startTime

def numba_energy(reps, fileName, return_new_buffers, swap_bytes):
    iterator = numpyinterface.iterate(
        fileName, "twoMuon", "px", "py", "pz", "mass_mumu",
        return_new_buffers = return_new_buffers,
        swap_bytes = swap_bytes)

    startTime = time.time()

    for i in xrange(reps):
        total = 0
        for start, end, px, py, pz, mass_mumu in iterator:
            total = energysum(px, py, pz, mass_mumu)

    endTime = time.time()

    return endTime - startTime

def rootnumpy_energy(reps, fileName):
    file = ROOT.TFile(fileName)
    tree = file.Get("twoMuon")

    startTime = time.time()

    for i in xrange(reps):
        array = root_numpy.tree2array(tree, branches=["px", "py", "pz", "mass_mumu"])
        px = array["px"]
        py = array["py"]
        pz = array["pz"]
        mass_mumu = array["mass_mumu"]
        total = numpy.sqrt(px**2 + py**2 + pz**2 + mass_mumu**2).sum()

    endTime = time.time()

    return endTime - startTime

WARM_UP = 5
REPS = 100

for fileName in "TrackResonanceNtuple_uncompressed.root", "TrackResonanceNtuple_compressed.root":
    for label, return_new_buffers in ("new buffers", True), ("views", False):
        print fileName, "momentum numpy big-endian", label,
        numpy_momentum(WARM_UP, fileName, return_new_buffers, False)     # warm up
        print numpy_momentum(REPS, fileName, return_new_buffers, False)  # real run

        print fileName, "momentum numpy little-endian", label,
        numpy_momentum(WARM_UP, fileName, return_new_buffers, True)      # warm up
        print numpy_momentum(REPS, fileName, return_new_buffers, True)   # real run

        print fileName, "momentum numba little-endian", label,
        numba_momentum(WARM_UP, fileName, return_new_buffers, True)      # warm up
        print numba_momentum(REPS, fileName, return_new_buffers, True)   # real run

    print fileName, "momentum root_numpy",
    rootnumpy_momentum(WARM_UP, fileName)                                # warm up
    print rootnumpy_momentum(REPS, fileName)                             # real run

print
print numpyinterface.performance()
print

for fileName in "TrackResonanceNtuple_uncompressed.root", "TrackResonanceNtuple_compressed.root":
    for label, return_new_buffers in ("new buffers", True), ("views", False):
        print fileName, "energy numpy big-endian", label,
        numpy_energy(WARM_UP, fileName, return_new_buffers, False)       # warm up
        print numpy_energy(REPS, fileName, return_new_buffers, False)    # real run

        print fileName, "energy numpy little-endian", label,
        numpy_energy(WARM_UP, fileName, return_new_buffers, True)        # warm up
        print numpy_energy(REPS, fileName, return_new_buffers, True)     # real run

        print fileName, "energy numba little-endian", label,
        numba_energy(WARM_UP, fileName, return_new_buffers, True)        # warm up
        print numba_energy(REPS, fileName, return_new_buffers, True)     # real run

    print fileName, "energy root_numpy",
    rootnumpy_energy(WARM_UP, fileName)                                  # warm up
    print rootnumpy_energy(REPS, fileName)                               # real run

print
print numpyinterface.performance()
print

