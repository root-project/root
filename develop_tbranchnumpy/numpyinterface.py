import numpy

import _numpyinterface
from _numpyinterface import iterate    # this is a high-level function, should be exposed at this level

# def dictofarrays(*args,
#                  allocate=lambda dtype, shape: numpy.empty(shape, dtype=dtype),
#                  trim=lambda array, length: array[:length],
#                  swap_bytes=True):
