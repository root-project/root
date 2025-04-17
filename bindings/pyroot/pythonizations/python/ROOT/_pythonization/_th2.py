# Author: Vincenzo Eduardo Padulano 12/2024

################################################################################
# Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from . import pythonization
from ROOT._pythonization._memory_utils import inject_constructor_releasing_ownership


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

    from ROOT._pythonization._uhi import _add_plotting_features

    # Add UHI plotting features
    pythonization(klass)(_add_plotting_features)
