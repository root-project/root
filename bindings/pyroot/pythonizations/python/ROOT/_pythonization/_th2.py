# Author: Vincenzo Eduardo Padulano 12/2024

################################################################################
# Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from ROOT._pythonization._memory_utils import inject_constructor_releasing_ownership

from . import pythonization


# Fill with array-like data
def _FillWithArrayTH2(self, *args):
    """
    Fill a histogram using array-like input.
    Parameters:
    - self: histogram
    - args: arguments to FillN
            If the first 2 arguments are array-like:
            - converts them to numpy arrays
            - fills the histogram with these arrays
            - optional third argument is weights array,
              if not provided, weights of 1 are used
            Otherwise:
            - Arguments are passed directly to the original FillN method
    Returns:
    - Result of FillN if array case is detected, otherwise result of Fill
    Raises:
    - ValueError: If x, y, and/or weights do not have matching lengths
    """
    # If there are less than 2 arguments, cannot do vectorized Fill
    if len(args) < 2:
        return self._Fill(*args)

    import numpy as np

    try:
        x = np.asanyarray(args[0], dtype=np.float64)
        y = np.asanyarray(args[1], dtype=np.float64)

        if len(x) != len(y):
            raise ValueError(f"Length mismatch: x length ({len(x)}) != y length ({len(y)})")

        n = len(x)
    except Exception:
        # Not convertible
        return self._Fill(*args)

    if len(args) >= 3 and args[2] is not None:
        weights = np.asanyarray(args[2], dtype=np.float64)
        if len(weights) != n:
            raise ValueError(f"Length mismatch: data length ({n}) != weights length ({len(weights)})")
    else:
        weights = np.ones(n)

    return self.FillN(n, x, y, weights)


# The constructors need to be pythonized for each derived class separately:
_th2_derived_classes_to_pythonize = [
    "TH2C",
    "TH2S",
    "TH2I",
    "TH2L",
    "TH2F",
    "TH2D",
    # "TH2Poly", # Derives from TH2 but does not automatically register
    # "TH2PolyBin", Does not derive from TH2
    "TProfile2D",
    # "TProfile2PolyBin", Derives from TH2PolyBin which does not derive from TH2
    "TProfile2Poly",
]

for klass in _th2_derived_classes_to_pythonize:
    pythonization(klass)(inject_constructor_releasing_ownership)

    from ROOT._pythonization._uhi.main import _add_plotting_features

    # Add UHI plotting features
    pythonization(klass)(_add_plotting_features)

    # Support vectorized Fill
    @pythonization(klass)
    def _enable_numpy_fill(klass):
        klass._Fill = klass.Fill
        klass.Fill = _FillWithArrayTH2
