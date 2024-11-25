# Author: Aaron Jomy CERN 09/2024
# Author: Vincenzo Eduardo Padulano CERN 09/2024

################################################################################
# Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

r"""
/**
\class TF1
\brief \parblock \endparblock
\htmlonly
<div class="pyrootbox">
\endhtmlonly
\anchor python
## PyROOT

The TF1 class has several additions for its use from Python, which are also
available in its subclasses TF2 and TF3.

First, TF1 instance can be initialized with user-defined Python functions. Given a generic Python callable,
the following can performed:

\code{.py}
def func(x: numpy.ndarray, pars: numpy.ndarray) -> float:
    return pars[0] * x[0] * x[0] + x[1] * pars[0]

my_func = ROOT.TF1("my_func", func, -10, 10, npar=2, ndim=2)
\endcode

Second, after performing the initialisation with a Python functor, the TF1 instance can be evaluated using the Pythonized
`TF1::EvalPar` function. The pythonization allows passing in 1D(single set of x variables) or 2D(a dataset) NumPy arrays.

The following example shows how we can create a TF1 instance with a Python function and evaluate it on a dataset:

\code{.py}
import ROOT
import math
import numpy as np

def pyf_tf1_coulomb(x, p):
    return p[1] * x[0] * x[1] / (p[0]**2) * math.exp(-p[2] / p[0])

rtf1_coulomb = ROOT.TF1("my_func", pyf_tf1_coulomb, -10, 10, ndims = 2, npars = 3)

# x dataset: 5 pairs of particle charges
x = np.array([
    [1.0, 10, 2.0],
    [1.5, 10, 2.5],
    [2.0, 10, 3.0],
    [2.5, 10, 3.5],
    [3.0, 10, 4.0]
])

params = np.array([
    [1.0],       # Distance between charges r
    [8.99e9],    # Coulomb constant k (in N·m²/C²)
    [0.1]        # Additional factor for modulation
])

# Slice to avoid the dummy column of 10's
res = rtf1_coulomb.EvalPar(x[:, ::2], params)
        
\endcode

\htmlonly
</div>
\endhtmlonly
*/
"""

from . import pythonization

def _TF1_EvalPar(self, vars, params):

    import ROOT
    import numpy

    x = numpy.ascontiguousarray(vars)

    if x.ndim == 1:
        return self._EvalPar(x, params)

    interface = x.__array_interface__
    shape = interface["shape"]

    nrows = shape[0]
    x_size = shape[1]

    if x_size > self.GetNdim():
        self.SetNdim(x_size)

    out = numpy.zeros(len(x))
    
    ROOT.Internal.EvalParMultiDim(self, out, x, x_size, nrows, params)
    return numpy.frombuffer(out, dtype=numpy.float64, count=nrows) 


def _TF1_Constructor(self, *args, **kwargs):
    """
    Forward the arguments to the C++ constructor and retain ownership. This
    helps avoiding double deletes due to ROOT automatic memory management.
    """
    self._cpp_constructor(*args, **kwargs)
    import ROOT
    ROOT.SetOwnership(self, False)


@pythonization('TF1')
def pythonize_tf1(klass):

    # Pythonizations for TH1::EvalPar
    klass._EvalPar = klass.EvalPar
    klass.EvalPar = _TF1_EvalPar

    klass._cpp_constructor = klass.__init__
    klass.__init__ = _TF1_Constructor
