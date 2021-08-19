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


class RooMsgService(object):
    """Some member functions of RooMsgService that take a RooCmdArg as argument also support keyword arguments.
    So far, this applies to RooMsgService::addStream.
    For example, the following code is equivalent in PyROOT:
    \code{.py}
    # Directly passing a RooCmdArg:
    ROOT.RooMsgService.instance().addStream(ROOT.RooFit.DEBUG, ROOT.RooFit.Topic(ROOT.RooFit.Tracing), ROOT.RooFit.ClassName("RooGaussian"))

    # With keyword arguments:
    ROOT.RooMsgService.instance().addStream(ROOT.RooFit.DEBUG, Topic = ROOT.RooFit.Tracing, ClassName = "RooGaussian")
    \endcode"""

    @cpp_signature(
        "Int_t RooMsgService::addStream(RooFit::MsgLevel level, const RooCmdArg& arg1=RooCmdArg(), const RooCmdArg& arg2=RooCmdArg(), const RooCmdArg& arg3=RooCmdArg(),"
        "    const RooCmdArg& arg4=RooCmdArg(), const RooCmdArg& arg5=RooCmdArg(), const RooCmdArg& arg6=RooCmdArg());"
    )
    def addStream(self, *args, **kwargs):
        """The RooMsgService::addStream() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArg of the function.
        """
        # Redefinition of `RooMsgService.addStream` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._addStream(*args, **kwargs)
