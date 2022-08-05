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


class RooAbsRealLValue(object):
    r"""Some member functions of RooAbsRealLValue that take a RooCmdArg as argument also support keyword arguments.
    So far, this applies to RooAbsRealLValue::createHistogram and RooAbsRealLValue::frame.
    For example, the following code is equivalent in PyROOT:
    \code{.py}
    # Directly passing a RooCmdArg:
    frame = x.frame(ROOT.RooFit.Name("xframe"), ROOT.RooFit.Title("RooPlot with decorations"), ROOT.RooFit.Bins(40))

    # With keyword arguments:
    frame = x.frame(Name="xframe", Title="RooPlot with decorations", Bins=40)
    \endcode
    """

    @cpp_signature(
        "TH1 *RooAbsRealLValue::createHistogram(const char *name,"
        "    const RooCmdArg& arg1=RooCmdArg::none(), const RooCmdArg& arg2=RooCmdArg::none(),"
        "    const RooCmdArg& arg3=RooCmdArg::none(), const RooCmdArg& arg4=RooCmdArg::none(),"
        "    const RooCmdArg& arg5=RooCmdArg::none(), const RooCmdArg& arg6=RooCmdArg::none(),"
        "    const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) const ;"
    )
    def createHistogram(self, *args, **kwargs):
        r"""The RooAbsRealLValue::createHistogram() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        # Redefinition of `RooAbsRealLValue.createHistogram` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._createHistogram(*args, **kwargs)

    @cpp_signature(
        "RooPlot *RooAbsRealLValue::frame(const RooCmdArg& arg1, const RooCmdArg& arg2=RooCmdArg::none(),"
        "    const RooCmdArg& arg3=RooCmdArg::none(), const RooCmdArg& arg4=RooCmdArg::none(), const RooCmdArg& arg5=RooCmdArg::none(),"
        "    const RooCmdArg& arg6=RooCmdArg::none(), const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) const ;"
    )
    def frame(self, *args, **kwargs):
        r"""The RooAbsRealLValue::frame() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        # Redefinition of `RooAbsRealLValue.frame` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        frame = self._frame(*args, **kwargs)

        # Add a Python reference to the plot variable to the RooPlot because
        # the plot variable needs to survive at least as long as the plot.
        frame._plotVar = self

        return frame
