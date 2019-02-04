# Author: Enric Tejedor CERN  02/2019

################################################################################
# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from ROOT import pythonization
from cppyy.gbl import TIter


# Item access

def _getitem_pyz(self, idx):
    # Parameters:
    # - self: collection to get item from
    # - idx: index of the item
    # Returns:
    # - self[idx]
    res = self.At(idx)

    if not res:
        raise IndexError('list index out of range')

    return res

def _setitem_pyz(self, idx, val):
    # Parameters:
    # - self: collection where to set item
    # - idx: index of the item
    # - val: value of the item
    _delitem_pyz(self, idx)

    self.AddAt(val, idx)

def _delitem_pyz(self, idx):
    # Parameters:
    # - self: collection to delete item from
    # - idx: index of the item
    res = self.RemoveAt(idx)

    if not res:
        raise IndexError('list assignment index out of range')


@pythonization()
def pythonize_tseqcollection(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: string containing the name of the class

    if name == 'TSeqCollection':
        klass.__getitem__ = _getitem_pyz
        klass.__setitem__ = _setitem_pyz
        klass.__delitem__ = _delitem_pyz

    return True
