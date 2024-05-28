# Author: Enric Tejedor CERN  02/2019

################################################################################
# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from . import pythonization


def _TClonesArray__setitem__(self, key, val):
    raise RuntimeError(
        "You can't set items of a TClonesArray! You can however TClonesArray::ConstructedAt() to access an element that is guaranteed to be constructed, and then modify it in-place."
    )


@pythonization("TClonesArray")
def pythonize_tclonesarray(klass):
    # Parameters:
    # klass: class to be pythonized

    # Add item setter method
    klass.__setitem__ = _TClonesArray__setitem__
