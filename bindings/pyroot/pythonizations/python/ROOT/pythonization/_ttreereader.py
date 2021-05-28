# Author: Enric Tejedor CERN  05/2021

################################################################################
# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from ROOT import pythonization


# Iteration
def _iter(self):
    # Parameters:
    # - self: TTreeReader object
    # Returns:
    # - A TTreeReader iterator for self
    while self.Next():
        yield self.GetCurrentEntry()


@pythonization()
def pythonize_ttreereader(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: name of the class

    if name == 'TTreeReader':
        klass.__iter__ = _iter
