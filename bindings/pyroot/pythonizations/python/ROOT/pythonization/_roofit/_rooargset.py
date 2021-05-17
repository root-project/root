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

import cppyy
import operator


class RooArgSet(RooAbsCollection):
    def __getitem__(self, key):

        # other than the RooArgList, the RooArgSet also supports string keys
        if isinstance(key, (str, cppyy.gbl.TString, cppyy.gbl.std.string)):
            return self._getitem(key)

        try:
            operator.index(key)
        except TypeError:
            raise TypeError("RooArgList indices must be integers or strings")

        # support for negative indexing
        if key < 0:
            key = key + len(self)

        if key < 0 or key >= len(self):
            raise IndexError("RooArgList index out of range")

        return self._getitem(key)
