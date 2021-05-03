# Authors:
# * Harshal Shende  03/2021
# * Jonas Rembser 03/2021

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################


from ._rooabsreal import RooAbsReal
from ._utils import _getter


class RooAbsPdf(RooAbsReal):
    def fitTo(self, *args, **kwargs):
        """
        Docstring
        """
        # Redefinition of `RooAbsPdf.fitTo` for keyword arguments.
        # the keywords must correspond to the CmdArg of the `fitTo` function.
        # Parameters:
        # self: instance of `RooAbsPdf` class
        # *args: arguments passed to `fitTo`
        # **kwargs: keyword arguments passed to `fitTo`
        if not kwargs:
            return self._fitTo(*args)
        else:
            nargs = args + tuple((_getter(k, v) for k, v in kwargs.items()))
            return self._fitTo(*nargs)

    def plotOn(self, *args, **kwargs):
        """
        Docstring
        """
        # Redefinition of `RooAbsReal.plotOn` for keyword arguments.
        # the keywords must correspond to the CmdArg of the `plotOn` function.
        # Parameters:
        # self: instance of `RooAbsReal` class
        # *args: arguments passed to `plotOn`
        # **kwargs: keyword arguments passed to `plotOn`
        if not kwargs:
            return self._plotOn(*args)
        else:
            nargs = args + tuple((_getter(k, v) for k, v in kwargs.items()))
            return self._plotOn(*nargs)
