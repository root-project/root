# Author: Enric Tejedor CERN  02/2019

################################################################################
# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from . import pythonization
from ROOT._pythonization._memory_utils import inject_constructor_releasing_ownership, inject_clone_releasing_ownership, _SetDirectory_SetOwnership

# Multiplication by constant

def _imul(self, c):
    # Parameters:
    # - self: histogram
    # - c: constant by which to multiply the histogram
    # Returns:
    # - A multiplied histogram (in place)
    self.Scale(c)
    return self


# The constructors need to be pythonized for each derived class separately:
_th1_derived_classes_to_pythonize = [
    "TH1C",
    "TH1S",
    "TH1I",
    "TH1L",
    "TH1F",
    "TH1D",
    "TH1K",
    "TProfile",
]

for klass in _th1_derived_classes_to_pythonize:
    pythonization(klass)(inject_constructor_releasing_ownership)


@pythonization('TH1')
def pythonize_th1(klass):
    # Parameters:
    # klass: class to be pythonized

    # Support hist *= scalar
    klass.__imul__ = _imul

    klass._Original_SetDirectory = klass.SetDirectory
    klass.SetDirectory = _SetDirectory_SetOwnership

    inject_clone_releasing_ownership(klass)
