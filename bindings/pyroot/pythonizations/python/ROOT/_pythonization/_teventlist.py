# Author: Vincenzo Eduardo Padulano 12/2024

################################################################################
# Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from . import pythonization
from ROOT._pythonization._memory_utils import _constructor_releasing_ownership, _SetDirectory_SetOwnership


@pythonization("TEventList")
def pythonize_tentrylist(klass):
    klass._cpp_constructor = klass.__init__
    klass.__init__ = _constructor_releasing_ownership

    klass._Original_SetDirectory = klass.SetDirectory
    klass.SetDirectory = _SetDirectory_SetOwnership
