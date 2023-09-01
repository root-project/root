# Authors:
# * Jonas Rembser 05/2021

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################


from ._rooabscollection import RooAbsCollection

import operator


class RooArgList(RooAbsCollection):
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
