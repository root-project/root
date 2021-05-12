# Authors:
# * Hinnerk C. Schmidt 02/2021
# * Jonas Rembser 03/2021

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################


r'''
/**
\class RooAbsReal
\brief \parblock \endparblock
\htmlonly
<div class="pyrootbox">
\endhtmlonly

## PyROOT

Some member functions of RooAbsReal that take a RooCmdArg as argument also support keyword arguments.
So far, this applies to RooAbsReal::plotOn.
For example, the following code is equivalent in PyROOT:
\code{.py}
# Directly passing a RooCmdArg:
var.plotOn(frame, ROOT.RooFit.Components("background"))

# With keyword arguments:
var.plotOn(frame, Components="background")
\endcode

\htmlonly
</div>
\endhtmlonly
*/
'''

from ._utils import _getter


class RooAbsReal(object):
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
