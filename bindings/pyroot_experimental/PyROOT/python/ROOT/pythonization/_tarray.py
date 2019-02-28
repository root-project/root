# Author: Enric Tejedor CERN  11/2018

################################################################################
# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from ROOT import pythonization

from ._generic import _add_getitem_checked


@pythonization()
def pythonize_tarray(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: string containing the name of the class

    if name == 'TArray':
        # Support `len(a)` as `a.GetSize()`
        klass.__len__ = klass.GetSize

    elif name.startswith('TArray'):
        # Add checked __getitem__. It has to be directly added to the TArray
        # subclasses, which have a default __getitem__.
        # The new __getitem__ allows to throw pythonic IndexError when index
        # is out of range and to iterate over the array.
        _add_getitem_checked(klass)

    return True
