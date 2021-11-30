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
    def __init__(self, *args, **kwargs):
        """Pythonization of RooArgSet constructor to support implicit
        conversion from Python sets.
        """
        # Note: This simple Pythonization caused me days of headache.
        # Initially, I was also checking of `len(kwargs) == 0`, but it just
        # didn't work. Eventually, I understood that when cppy attempts
        # implicit conversion, a magic `__cppyy_no_implicit=True` keyword
        # argument is added, hence the `len(kwargs) == 0` check breaks the
        # implicit conversion!
        if len(args) == 1 and isinstance(args[0], set):
            return self._init(*args[0], **kwargs)
        return self._init(*args, **kwargs)

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
