# Author: Vincenzo Eduardo Padulano 12/2024

################################################################################
# Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from . import pythonization


def _SetDirectory_SetOwnership(self, dir):
    self._Original_SetDirectory(dir)
    if dir:
        # If we are actually registering with a directory, give ownership to C++
        import ROOT
        ROOT.SetOwnership(self, False)


@pythonization("TEfficiency")
def pythonize_tefficiency(klass):

    klass._Original_SetDirectory = klass.SetDirectory
    klass.SetDirectory = _SetDirectory_SetOwnership
