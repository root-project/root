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
\class RooMCStudy
\brief \parblock \endparblock
\htmlonly
<div class="pyrootbox">
\endhtmlonly

## PyROOT

Some member functions of RooMCStudy that take a RooCmdArg as argument also support keyword arguments.
So far, this applies to constructor RooMCStudy(), RooMCStudy::plotParamOn, RooMCStudy::plotParam, RooMCStudy::plotNLL, RooMCStudy::plotError and RooMCStudy::plotPull.
For example, the following code is equivalent in PyROOT:
\code{.py}
# Directly passing a RooCmdArg:
frame3 = mcstudy.plotPull(mean, ROOT.RooFit.Bins(40), ROOT.RooFit.FitGauss(True))

# With keyword arguments:
frame3 = mcstudy.plotPull(mean, Bins=40, FitGauss=True)
\endcode

\htmlonly
</div>
\endhtmlonly
*/
"""

from ._utils import _kwargs_to_roocmdargs


class RooMCStudy(object):
    def __init__(self, *args, **kwargs):
        # Redefinition of `RooMCStudy` constructor for keyword arguments.
        # The keywords must correspond to the CmdArg of the constructor function.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        self._init(*args, **kwargs)

    def plotParamOn(self, *args, **kwargs):
        # Redefinition of `RooMCStudy.plotParamOn` for keyword arguments.
        # The keywords must correspond to the CmdArg of the `plotParamOn` function.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._plotParamOn(*args, **kwargs)

    def plotParam(self, *args, **kwargs):
        # Redefinition of `RooMCStudy.plotParam` for keyword arguments.
        # The keywords must correspond to the CmdArg of the `plotParam` function.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._plotParam(*args, **kwargs)

    def plotNLL(self, *args, **kwargs):
        # Redefinition of `RooMCStudy.plotNLL` for keyword arguments.
        # The keywords must correspond to the CmdArg of the `plotNLL` function.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._plotNLL(*args, **kwargs)

    def plotError(self, *args, **kwargs):
        # Redefinition of `RooMCStudy.plotError` for keyword arguments.
        # The keywords must correspond to the CmdArg of the `plotError` function.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._plotError(*args, **kwargs)

    def plotPull(self, *args, **kwargs):
        # Redefinition of `RooMCStudy.plotPull` for keyword arguments.
        # The keywords must correspond to the CmdArg of the `plotPull` function.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._plotPull(*args, **kwargs)
