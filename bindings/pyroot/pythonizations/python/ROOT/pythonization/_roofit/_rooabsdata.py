# Authors:
# * Hinnerk C. Schmidt 02/2021
# * Jonas Rembser 03/2021
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
\class RooAbsData
\brief \parblock \endparblock
\htmlonly
<div class="pyrootbox">
\endhtmlonly

## PyROOT

Some member functions of RooAbsData that take a RooCmdArg as argument also support keyword arguments.
This applies to RooAbsData::plotOn, RooAbsData::createHistogram, RooAbsData::reduce, RooAbsData::statOn.
For example, the following code is equivalent in PyROOT:
\code{.py}
# Directly passing a RooCmdArg:
data.plotOn(frame, ROOT.RooFit.CutRange("r1"))

# With keyword arguments:
data.plotOn(frame, CutRange="r1")
\endcode

\htmlonly
</div>
\endhtmlonly
*/
"""

from ._utils import _kwargs_to_roocmdargs


class RooAbsData(object):
    def plotOn(self, *args, **kwargs):
        # Redefinition of `RooAbsData.plotOn` for keyword arguments.
        # The keywords must correspond to the CmdArg of the `plotOn` function.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._plotOn(*args, **kwargs)

    def createHistogram(self, *args, **kwargs):
        # Redefinition of `RooAbsData.createHistogram` for keyword arguments.
        # The keywords must correspond to the CmdArg of the `createHistogram` function.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._createHistogram(*args, **kwargs)

    def reduce(self, *args, **kwargs):
        # Redefinition of `RooAbsData.reduce` for keyword arguments.
        # The keywords must correspond to the CmdArg of the `reduce` function.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._reduce(*args, **kwargs)

    def statOn(self, *args, **kwargs):
        # Redefinition of `RooAbsData.statOn` for keyword arguments.
        # The keywords must correspond to the CmdArg of the `statOn` function.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._statOn(*args, **kwargs)
