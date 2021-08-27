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


from ._utils import _kwargs_to_roocmdargs, cpp_signature


class RooMCStudy(object):
    r"""Some member functions of RooMCStudy that take a RooCmdArg as argument also support keyword arguments.
    So far, this applies to constructor RooMCStudy(), RooMCStudy::plotParamOn, RooMCStudy::plotParam, RooMCStudy::plotNLL, RooMCStudy::plotError and RooMCStudy::plotPull.
    For example, the following code is equivalent in PyROOT:
    \code{.py}
    # Directly passing a RooCmdArg:
    frame3 = mcstudy.plotPull(mean, ROOT.RooFit.Bins(40), ROOT.RooFit.FitGauss(True))

    # With keyword arguments:
    frame3 = mcstudy.plotPull(mean, Bins=40, FitGauss=True)
    \endcode
    """

    @cpp_signature(
        "RooMCStudy(const RooAbsPdf& model, const RooArgSet& observables,"
        "    const RooCmdArg& arg1=RooCmdArg::none(), const RooCmdArg& arg2=RooCmdArg::none(),"
        "    const RooCmdArg& arg3=RooCmdArg::none(), const RooCmdArg& arg4=RooCmdArg::none(), const RooCmdArg& arg5=RooCmdArg::none(),"
        "    const RooCmdArg& arg6=RooCmdArg::none(), const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) ;"
    )
    def __init__(self, *args, **kwargs):
        r"""The RooMCStudy constructor is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArg of the constructor function.
        """
        # Redefinition of `RooMCStudy` constructor for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        self._init(*args, **kwargs)

    @cpp_signature(
        "RooPlot *RooMCStudy::plotParamOn(RooPlot* frame, const RooCmdArg& arg1=RooCmdArg::none(), const RooCmdArg& arg2=RooCmdArg::none(),"
        "    const RooCmdArg& arg3=RooCmdArg::none(), const RooCmdArg& arg4=RooCmdArg::none(),"
        "    const RooCmdArg& arg5=RooCmdArg::none(), const RooCmdArg& arg6=RooCmdArg::none(),"
        "    const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) ;"
    )
    def plotParamOn(self, *args, **kwargs):
        r"""The RooMCStudy::plotParamOn() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArg of the function."""
        # Redefinition of `RooMCStudy.plotParamOn` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._plotParamOn(*args, **kwargs)

    @cpp_signature(
        [
            "RooPlot *RooMCStudy::plotParam(const RooRealVar& param, const RooCmdArg& arg1=RooCmdArg::none(), const RooCmdArg& arg2=RooCmdArg::none(),"
            "    const RooCmdArg& arg3=RooCmdArg::none(), const RooCmdArg& arg4=RooCmdArg::none(),"
            "    const RooCmdArg& arg5=RooCmdArg::none(), const RooCmdArg& arg6=RooCmdArg::none(),"
            "    const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) ;",
            "RooPlot *RooMCStudy::plotParam(const char* paramName, const RooCmdArg& arg1=RooCmdArg::none(), const RooCmdArg& arg2=RooCmdArg::none(),"
            "    const RooCmdArg& arg3=RooCmdArg::none(), const RooCmdArg& arg4=RooCmdArg::none(), const RooCmdArg& arg5=RooCmdArg::none(), "
            "    const RooCmdArg& arg6=RooCmdArg::none(), const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) ;",
        ]
    )
    def plotParam(self, *args, **kwargs):
        r"""The RooMCStudy::plotParam() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArg of the function.
        """
        # Redefinition of `RooMCStudy.plotParam` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._plotParam(*args, **kwargs)

    @cpp_signature(
        "RooPlot *RooMCStudy::plotNLL(const RooCmdArg& arg1=RooCmdArg::none(), const RooCmdArg& arg2=RooCmdArg::none(),"
        "    const RooCmdArg& arg3=RooCmdArg::none(), const RooCmdArg& arg4=RooCmdArg::none(),"
        "    const RooCmdArg& arg5=RooCmdArg::none(), const RooCmdArg& arg6=RooCmdArg::none(),"
        "    const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) ;"
    )
    def plotNLL(self, *args, **kwargs):
        r"""The RooMCStudy::plotNLL() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArg of the function.
        """
        # Redefinition of `RooMCStudy.plotNLL` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._plotNLL(*args, **kwargs)

    @cpp_signature(
        "RooPlot *RooMCStudy::plotError(const RooRealVar& param, const RooCmdArg& arg1=RooCmdArg::none(), const RooCmdArg& arg2=RooCmdArg::none(),"
        "    const RooCmdArg& arg3=RooCmdArg::none(), const RooCmdArg& arg4=RooCmdArg::none(),"
        "    const RooCmdArg& arg5=RooCmdArg::none(), const RooCmdArg& arg6=RooCmdArg::none(),"
        "    const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) ;"
    )
    def plotError(self, *args, **kwargs):
        r"""The RooMCStudy::plotError() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArg of the function.
        """
        # Redefinition of `RooMCStudy.plotError` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._plotError(*args, **kwargs)

    @cpp_signature(
        "RooPlot *RooMCStudy::plotPull(const RooRealVar& param, const RooCmdArg& arg1, const RooCmdArg& arg2=RooCmdArg::none(),"
        "    const RooCmdArg& arg3=RooCmdArg::none(), const RooCmdArg& arg4=RooCmdArg::none(),"
        "    const RooCmdArg& arg5=RooCmdArg::none(), const RooCmdArg& arg6=RooCmdArg::none(),"
        "    const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) ;"
    )
    def plotPull(self, *args, **kwargs):
        r"""The RooMCStudy::plotError() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArg of the function.
        """
        # Redefinition of `RooMCStudy.plotPull` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._plotPull(*args, **kwargs)
