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
\class RooAbsReal
\brief \parblock \endparblock
\htmlonly
<div class="pyrootbox">
\endhtmlonly

## PyROOT

Some member functions of RooAbsReal that take a RooCmdArg as argument also support keyword arguments.
So far, this applies to RooAbsReal::plotOn, RooAbsReal::createHistogram, RooAbsReal::chi2FitTo,
RooAbsReal::createChi2, RooAbsReal::createRunningIntegral and RooAbsReal::createIntegral
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
"""

from ._utils import _kwargs_to_roocmdargs


class RooAbsReal(object):
    def plotOn(self, *args, **kwargs):
        # Redefinition of `RooAbsReal.plotOn` for keyword arguments.
        # The keywords must correspond to the CmdArg of the `plotOn` function.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._plotOn(*args, **kwargs)

    def createHistogram(self, *args, **kwargs):
        # Redefinition of `RooAbsReal.createHistogram` for keyword arguments.
        # The keywords must correspond to the CmdArg of the `createHistogram` function.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._createHistogram(*args, **kwargs)

    def createIntegral(self, *args, **kwargs):
        # Redefinition of `RooAbsReal.createIntegral` for keyword arguments.
        # The keywords must correspond to the CmdArg of the `createIntegral` function.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._createIntegral(*args, **kwargs)

    def createRunningIntegral(self, *args, **kwargs):
        # Redefinition of `RooAbsReal.createRunningIntegral` for keyword arguments.
        # The keywords must correspond to the CmdArg of the `createRunningIntegral` function.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._createRunningIntegral(*args, **kwargs)

    def createChi2(self, *args, **kwargs):
        # Redefinition of `RooAbsReal.createChi2` for keyword arguments.
        # The keywords must correspond to the CmdArg of the `createChi2` function.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._createChi2(*args, **kwargs)

    def chi2FitTo(self, *args, **kwargs):
        # Redefinition of `RooAbsReal.chi2FitTo` for keyword arguments.
        # The keywords must correspond to the CmdArg of the `chi2FitTo` function.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._chi2FitTo(*args, **kwargs)
