# Authors:
# * Jonas Rembser 06/2021
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
\class RooDataSet
\brief \parblock \endparblock
\htmlonly
<div class="pyrootbox">
\endhtmlonly

## PyROOT

Some member functions of RooDataSet that take a RooCmdArg as argument also support keyword arguments.
So far, this applies to RooDataSet() constructor and RooDataSet::plotOnXY.
For example, the following code is equivalent in PyROOT:
\code{.py}
# Directly passing a RooCmdArg:
dxy = ROOT.RooDataSet("dxy", "dxy", ROOT.RooArgSet(x, y), ROOT.RooFit.StoreError(ROOT.RooArgSet(x, y)))

# With keyword arguments:
dxy = ROOT.RooDataSet("dxy", "dxy", ROOT.RooArgSet(x, y), StoreError=(ROOT.RooArgSet(x, y)))
\endcode

\htmlonly
</div>
\endhtmlonly
*/
"""

from ._utils import _kwargs_to_roocmdargs


class RooDataSet(object):
    def __init__(self, *args, **kwargs):
        # Redefinition of `RooDataSet` constructor for keyword arguments.
        # The keywords must correspond to the CmdArg of the constructor function.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        self._init(*args, **kwargs)

    def plotOnXY(self, *args, **kwargs):
        # Redefinition of `RooDataSet.plotOnXY` for keyword arguments.
        # The keywords must correspond to the CmdArg of the `plotOnXY` function.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._plotOnXY(*args, **kwargs)
