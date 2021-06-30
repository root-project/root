# Authors:
# * Jonas Rembser 05/2021
# * Harshal Shende 06/2021

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################


r"""
/**
\class RooAbsCollection
\brief \parblock \endparblock
\htmlonly
<div class="pyrootbox">
\endhtmlonly

## PyROOT

Some member functions of RooAbsCollection that take a RooCmdArg as argument also support keyword arguments.
So far, this applies to RooAbsCollection::printLatex.
For example, the following code is equivalent in PyROOT:
\code{.py}
# Directly passing a RooCmdArg:
params.printLatex(ROOT.RooFit.Sibling(initParams), ROOT.RooFit.Columns(2))

# With keyword arguments:
params.printLatex(Sibling=initParams, Columns =2)

\endcode

\htmlonly
</div>
\endhtmlonly
*/
"""

from ._utils import _kwargs_to_roocmdargs
from libcppyy import SetOwnership


class RooAbsCollection(object):
    def addClone(self, arg, silent=False):
        clonedArg = self._addClone(arg, silent)
        SetOwnership(clonedArg, False)

    def addOwned(self, arg, silent=False):
        self._addOwned(arg, silent)
        SetOwnership(arg, False)

    def printLatex(self, *args, **kwargs):
        # Redefinition of `RooAbsCollection.printLatex` for keyword arguments.
        # The keywords must correspond to the CmdArg of the `printLatex` function.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._printLatex(*args, **kwargs)
