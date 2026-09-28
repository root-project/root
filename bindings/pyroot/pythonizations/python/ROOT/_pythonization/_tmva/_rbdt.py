# Author: Stefan Wunsch CERN  09/2019

################################################################################
# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from .. import pythonization

# Whether the JIT batch-inference helper has been declared to Cling
_rbdt_batch_helper_declared = False


def _rbdt_batch_compute(model, x, cols):
    # Compute the batch inference on the flat, row-major numpy array x with the
    # given number of columns per event. Going through a small JIT-compiled
    # helper here avoids passing std::span as a function argument through
    # cppyy: on builds with a C++17 standard setting, std::span comes from
    # ROOT's own backport in ROOT/span.hxx as std::__ROOT::span, which cppyy
    # does not know how to convert numpy arrays to.
    import ROOT

    global _rbdt_batch_helper_declared
    if not _rbdt_batch_helper_declared:
        ROOT.gInterpreter.Declare(
            r"""
namespace TMVA {
namespace Experimental {
namespace Internal {
std::vector<float> RBDTBatchCompute(RBDT const &model, float const *data, std::size_t size, unsigned int cols)
{
   return model.Compute(std::span<const float>{data, size}, cols);
}
} // namespace Internal
} // namespace Experimental
} // namespace TMVA
"""
        )
        _rbdt_batch_helper_declared = True

    return ROOT.TMVA.Experimental.Internal.RBDTBatchCompute(
        model, x.reshape(-1), x.shape[0] * x.shape[1], cols)


def Compute(self, x):
    import numpy as np

    import ROOT

    # numpy.array is a factory and the actual type of a numpy array is numpy.ndarray
    if isinstance(x, np.ndarray):
        if len(x.shape) == 1:
            x_ = ROOT.VecOps.AsRVec(x)
            y = self._OriginalCompute(x_)
            return np.asarray(y)
        elif len(x.shape) == 2:
            if x.dtype != np.float32:
                raise Exception("Call to Compute with a rank-2 numpy array requires float32 data type.")
            x_ = np.ascontiguousarray(x)
            y = _rbdt_batch_compute(self, x_, x_.shape[1])
            return np.asarray(y).reshape(x.shape[0], -1)
        else:
            raise Exception("Call to Compute can process only numpy arrays of rank 1 or 2.")

    # As fall-through we go to the original compute function and use the error-handling from cppyy
    return self._OriginalCompute(x)


@pythonization("RBDT", ns="TMVA::Experimental", is_prefix=True)
def pythonize_rbdt(klass):
    # Parameters:
    # klass: class to be pythonized

    klass._OriginalCompute = klass.Compute
    klass.Compute = Compute
