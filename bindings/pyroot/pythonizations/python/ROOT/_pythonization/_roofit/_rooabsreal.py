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


from ._utils import _kwargs_to_roocmdargs, cpp_signature


class RooAbsReal(object):
    r"""Some member functions of RooAbsReal that take a RooCmdArg as argument also support keyword arguments.
    So far, this applies to RooAbsReal::plotOn, RooAbsReal::createHistogram, RooAbsReal::chi2FitTo,
    RooAbsReal::createChi2, RooAbsReal::createRunningIntegral and RooAbsReal::createIntegral
    For example, the following code is equivalent in PyROOT:
    \code{.py}
    # Directly passing a RooCmdArg:
    var.plotOn(frame, ROOT.RooFit.Components("background"))

    # With keyword arguments:
    var.plotOn(frame, Components="background")
    \endcode
    """

    @cpp_signature(
        "RooPlot* RooAbsReal::plotOn(RooPlot* frame,"
        "    const RooCmdArg& arg1={}, const RooCmdArg& arg2={},"
        "    const RooCmdArg& arg3={}, const RooCmdArg& arg4={},"
        "    const RooCmdArg& arg5={}, const RooCmdArg& arg6={},"
        "    const RooCmdArg& arg7={}, const RooCmdArg& arg8={},"
        "    const RooCmdArg& arg9={}, const RooCmdArg& arg10={}) const ;"
    )
    def plotOn(self, *args, **kwargs):
        r"""The RooAbsReal::plotOn() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        # Redefinition of `RooAbsReal.plotOn` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._plotOn(*args, **kwargs)

    @cpp_signature(
        "TH1 *RooAbsReal::createHistogram(const char *name, const RooAbsRealLValue& xvar,"
        "    const RooCmdArg& arg1={}, const RooCmdArg& arg2={},"
        "    const RooCmdArg& arg3={}, const RooCmdArg& arg4={},"
        "    const RooCmdArg& arg5={}, const RooCmdArg& arg6={},"
        "    const RooCmdArg& arg7={}, const RooCmdArg& arg8={}) const ;"
    )
    def createHistogram(self, *args, **kwargs):
        r"""The RooAbsReal::createHistogram() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        # Redefinition of `RooAbsReal.createHistogram` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._createHistogram(*args, **kwargs)

    @cpp_signature(
        "RooAbsReal* RooAbsReal::createIntegral(const RooArgSet& iset, const RooCmdArg& arg1, const RooCmdArg& arg2={},"
        "    const RooCmdArg& arg3={}, const RooCmdArg& arg4={},"
        "    const RooCmdArg& arg5={}, const RooCmdArg& arg6={},"
        "    const RooCmdArg& arg7={}, const RooCmdArg& arg8={}) const ;"
    )
    def createIntegral(self, *args, **kwargs):
        r"""The RooAbsReal::createIntegral() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        # Redefinition of `RooAbsReal.createIntegral` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._createIntegral(*args, **kwargs)

    @cpp_signature(
        "RooAbsReal* RooAbsReal::createRunningIntegral(const RooArgSet& iset, const RooCmdArg& arg1, const RooCmdArg& arg2={},"
        "    const RooCmdArg& arg3={}, const RooCmdArg& arg4={},"
        "    const RooCmdArg& arg5={}, const RooCmdArg& arg6={},"
        "    const RooCmdArg& arg7={}, const RooCmdArg& arg8={}) ;"
    )
    def createRunningIntegral(self, *args, **kwargs):
        r"""The RooAbsReal::createRunningIntegral() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        # Redefinition of `RooAbsReal.createRunningIntegral` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._createRunningIntegral(*args, **kwargs)

    @cpp_signature(
        "RooAbsReal* RooAbsReal::createChi2(RooDataHist& data, const RooCmdArg& arg1={},  const RooCmdArg& arg2={},"
        "    const RooCmdArg& arg3={},  const RooCmdArg& arg4={}, const RooCmdArg& arg5={},"
        "    const RooCmdArg& arg6={},  const RooCmdArg& arg7={}, const RooCmdArg& arg8={}) ;"
    )
    def createChi2(self, *args, **kwargs):
        r"""The RooAbsReal::createChi2() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        # Redefinition of `RooAbsReal.createChi2` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._createChi2(*args, **kwargs)

    @cpp_signature(
        "RooFitResult *RooAbsReal::chi2FitTo(RooDataSet& xydata, const RooCmdArg& arg1={},  const RooCmdArg& arg2={},"
        "    const RooCmdArg& arg3={},  const RooCmdArg& arg4={}, const RooCmdArg& arg5={},"
        "    const RooCmdArg& arg6={},  const RooCmdArg& arg7={}, const RooCmdArg& arg8={}) ;"
    )
    def chi2FitTo(self, *args, **kwargs):
        r"""The RooAbsReal::chi2FitTo() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        # Redefinition of `RooAbsReal.chi2FitTo` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._chi2FitTo(*args, **kwargs)

    def getVal(self, normalizationSet=None):
        # We do the conversion to RooArgSet now, such that we can keep alive
        # the normalization set by setting it as an attribute of this
        # RooAbsReal.
        if isinstance(normalizationSet, (set, list, tuple)):
            import ROOT

            normalizationSet = ROOT.RooArgSet(normalizationSet)
        # With the pythonizations, we have the opportunity to use the Python
        # reference counting to make sure the last normalization set doesn't
        # get deleted under our feet (RooFit tries to use it by pointer when
        # you call getVal() without any normalization set the next time).
        if normalizationSet:
            self._getVal_normSet = normalizationSet
        return self._getVal(normalizationSet) if normalizationSet else self._getVal()
