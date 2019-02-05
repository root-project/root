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
from libcppyy import SetOwnership


# Item access

def _getitem_pyz(self, idx):
    # Parameters:
    # - self: collection to get the item/s from
    # - idx: index/slice of the item/s
    # Returns:
    # - self[idx]

    # Slice
    if isinstance(idx, slice):
        res = self.__class__()
        indices = idx.indices(len(self))
        for i in range(*indices):
            res.Add(self.At(i))
    # Number
    else:
        res = self.At(idx)

        if not res:
            raise IndexError('list index out of range')

    return res

def _setitem_pyz(self, idx, val):
    # Parameters:
    # - self: collection where to set item/s
    # - idx: index/slice of the item/s
    # - val: value to assign

    # Slice
    if isinstance(idx, slice):
        # The value we assign has to be iterable
        try:
            _ = iter(val)
        except TypeError:
            raise TypeError('can only assign an iterable')

        indices = idx.indices(len(self))
        step = indices[2]
        if step == 0:
            raise ValueError('slice step cannot be zero')

        rg = range(*indices)
        for elem in val:
            # Prevent this new Python proxy from owning the C++ object
            # Otherwise we get an 'already deleted' error in
            # TList::Clear when the application ends
            SetOwnership(elem, False)
            try:
                i = rg.pop(0)
                self[i] = elem
            except IndexError:
                # Empty range, just append
                self.append(elem)
    # Number
    else:
        _delitem_pyz(self, idx)
        self.AddAt(val, idx)

def _delitem_pyz(self, idx):
    # Parameters:
    # - self: collection to delete item from
    # - idx: index of the item

    # Slice
    if isinstance(idx, slice):
        indices = idx.indices(len(self))

        step = indices[2]
        if step == 0:
            raise ValueError('slice step cannot be zero')

        rg = range(*indices)

        if step > 0:
            # Need to remove starting from the end
            rg = reversed(rg)

        for i in rg:
            self.RemoveAt(i)
    # Number
    else:
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
