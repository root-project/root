# Author: Enric Tejedor CERN  11/2018

################################################################################
# Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from ROOT import pythonization

from ._generic import add_len

@pythonization()
def pythonize_rooabscollection(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: string containing the name of the class

    if name == 'RooAbsCollection':
        # Support `len(c)` as `c.getSize()`
        add_len(klass, 'getSize')
