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
\class RooAbsRealLValue
\brief \parblock \endparblock
\htmlonly
<div class="pyrootbox">
\endhtmlonly

## PyROOT

Some member functions of RooAbsRealLValue that take a RooCmdArg as argument also support keyword arguments.
So far, this applies to RooAbsRealLValue::createHistogram and RooAbsRealLValue::frame.
For example, the following code is equivalent in PyROOT:
\code{.py}
# Directly passing a RooCmdArg:
frame = x.frame(ROOT.RooFit.Name("xframe"), ROOT.RooFit.Title("RooPlot with decorations"), ROOT.RooFit.Bins(40))

# With keyword arguments:
frame = x.frame(Name="xframe", Title="RooPlot with decorations", Bins=40)
\endcode

\htmlonly
</div>
\endhtmlonly
*/
"""

from ._utils import _kwargs_to_roocmdargs


class RooAbsRealLValue(object):
    def createHistogram(self, *args, **kwargs):
        # Redefinition of `RooAbsRealLValue.createHistogram` for keyword arguments.
        # the keywords must correspond to the CmdArg of the `createHistogram` function.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._createHistogram(*args, **kwargs)

    def frame(self, *args, **kwargs):
        # Redefinition of `RooAbsRealLValue.frame` for keyword arguments.
        # the keywords must correspond to the CmdArg of the `frame` function.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._frame(*args, **kwargs)
