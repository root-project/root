# Authors:
# * Jonas Rembser 05/2021

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################


import operator

from ._rooabscollection import RooAbsCollection
from ._utils import _roo_collection_elements


class RooArgList(RooAbsCollection):

    __cpp_name__ = 'RooArgList'

    def __init__(self, *args, **kwargs):
        """Pythonization of RooArgList constructor to support implicit
        conversion from Python lists and tuples.
        """
        # A Python list is not a braced-init-list for the overload resolution,
        # so it does not convert to the C++ constructors taking an arbitrary
        # number of elements. Unpack it into the variadic constructor instead,
        # as `RooArgList({a, b})` would do in C++. This is also what the
        # implicit conversion of a list to `const RooArgList&` goes through.
        if len(args) >= 1 and isinstance(args[0], (list, tuple)):
            self._init(*_roo_collection_elements(args[0]), *args[1:], **kwargs)
            return
        self._init(*args, **kwargs)

    def __getitem__(self, key):
        try:
            operator.index(key)
        except TypeError:
            raise TypeError("RooArgList indices must be integers")

        # support for negative indexing
        if key < 0:
            key = key + len(self)

        if key < 0 or key >= len(self):
            raise IndexError("RooArgList index out of range")

        return self._getitem(key)
