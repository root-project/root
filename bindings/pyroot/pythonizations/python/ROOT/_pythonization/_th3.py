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
_th3_derived_classes_to_pythonize = [
    # "TGLTH3Composition", Derives from TH3 but does not automatically register
    "TH3C",
    "TH3S",
    "TH3I",
    "TH3L",
    "TH3F",
    "TH3D",
    "TProfile3D",
]

for klass in _th3_derived_classes_to_pythonize:
    pythonization(klass)(inject_constructor_releasing_ownership)
