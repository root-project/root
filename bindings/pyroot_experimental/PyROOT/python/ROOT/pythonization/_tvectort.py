# Author: Enric Tejedor CERN  02/2019

################################################################################
# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from ROOT import pythonization


@pythonization()
def pythonize_tvectort(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: string containing the name of the class

    if name.startswith('TVectorT'):
        # Support `len(v)` as `v.GetNoElements()`
        klass.__len__ = klass.GetNoElements

    return True
