# Author: Harshal Shende CERN  09/2022

################################################################################
# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################


def _func_name_orig(name):
    return name.replace("_", "")


def _numpy_content(buffer, args):
    # Helper to create a numpy array from a raw array pointer.
    #
    # Note: The data is copied.
    #
    # Args:
    #     buffer (cppyy.LowLevelView):
    #         The pointer to the beginning of the array data, usually
    #         obtained from a C++ function that returns a `double *`.
    #
    # Returns:
    #     numpy.ndarray
    import numpy as np

    return np.copy(np.frombuffer(buffer)) if isinstance(args, (np.ndarray, np.generic)) else buffer


def _numpy_getter(_dtype_dict, *args, **kwargs):
    # Parameters:
    # - args: Arguments
    # - kwargs: Keyword arguments
    # Returns:
    # - TH1/TH2/TH3 method according to datatype of the numpy array
    # - numpy arrays passed as arguments
    import ROOT
    import numpy as np

    npargs = tuple(arg if isinstance(arg, (np.ndarray, np.generic)) else "" for arg in args)
    if kwargs:
        args = tuple(kwargs.values())

    func = getattr(ROOT, _dtype_dict[str(npargs[0].dtype)])
    return func(*args), npargs
