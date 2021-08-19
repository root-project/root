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


class RooAbsData(object):
    """Some member functions of RooAbsData that take a RooCmdArg as argument also support keyword arguments.
    This applies to RooAbsData::plotOn, RooAbsData::createHistogram, RooAbsData::reduce, RooAbsData::statOn.
    For example, the following code is equivalent in PyROOT:
    \code{.py}
    # Directly passing a RooCmdArg:
    data.plotOn(frame, ROOT.RooFit.CutRange("r1"))

    # With keyword arguments:
    data.plotOn(frame, CutRange="r1")
    \endcode
    """

    @cpp_signature(
        "RooPlot *RooAbsData::plotOn(RooPlot* frame,"
        "			  const RooCmdArg& arg1=RooCmdArg::none(), const RooCmdArg& arg2=RooCmdArg::none(),"
        "			  const RooCmdArg& arg3=RooCmdArg::none(), const RooCmdArg& arg4=RooCmdArg::none(),"
        "			  const RooCmdArg& arg5=RooCmdArg::none(), const RooCmdArg& arg6=RooCmdArg::none(),"
        "			  const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) const ;"
    )
    def plotOn(self, *args, **kwargs):
        """The RooAbsData::plotOn() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        # Redefinition of `RooAbsData.plotOn` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._plotOn(*args, **kwargs)

    @cpp_signature(
        "TH1 *RooAbsData::createHistogram(const char *name, const RooAbsRealLValue& xvar,"
        "                       const RooCmdArg& arg1=RooCmdArg::none(), const RooCmdArg& arg2=RooCmdArg::none(),"
        "                       const RooCmdArg& arg3=RooCmdArg::none(), const RooCmdArg& arg4=RooCmdArg::none(),"
        "                       const RooCmdArg& arg5=RooCmdArg::none(), const RooCmdArg& arg6=RooCmdArg::none(),"
        "                       const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) const ;"
    )
    def createHistogram(self, *args, **kwargs):
        """The RooAbsData::createHistogram() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        # Redefinition of `RooAbsData.createHistogram` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._createHistogram(*args, **kwargs)

    @cpp_signature(
        "RooAbsData *RooAbsData::reduce(const RooCmdArg& arg1,const RooCmdArg& arg2=RooCmdArg(),"
        "                   const RooCmdArg& arg3=RooCmdArg(),const RooCmdArg& arg4=RooCmdArg(),"
        "                   const RooCmdArg& arg5=RooCmdArg(),const RooCmdArg& arg6=RooCmdArg(),"
        "                   const RooCmdArg& arg7=RooCmdArg(),const RooCmdArg& arg8=RooCmdArg()) ;"
    )
    def reduce(self, *args, **kwargs):
        """The RooAbsData::reduce() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        # Redefinition of `RooAbsData.reduce` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._reduce(*args, **kwargs)

    @cpp_signature(
        "RooPlot *RooAbsData::statOn(RooPlot* frame,"
        "                          const RooCmdArg& arg1=RooCmdArg::none(), const RooCmdArg& arg2=RooCmdArg::none(),"
        "                          const RooCmdArg& arg3=RooCmdArg::none(), const RooCmdArg& arg4=RooCmdArg::none(),"
        "                          const RooCmdArg& arg5=RooCmdArg::none(), const RooCmdArg& arg6=RooCmdArg::none(),"
        "                          const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) ;"
    )
    def statOn(self, *args, **kwargs):
        """The RooAbsData::statOn() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        # Redefinition of `RooAbsData.statOn` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._statOn(*args, **kwargs)
