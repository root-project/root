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


class RooChi2Var(object):
    """Constructor of RooChi2Var takes a RooCmdArg as argument also supports keyword arguments."""

    @cpp_signature(
        [
            "RooChi2Var(const char* name, const char* title, RooAbsReal& func, RooDataHist& data,"
            "        const RooCmdArg& arg1, const RooCmdArg& arg2=RooCmdArg::none(),const RooCmdArg& arg3=RooCmdArg::none(),"
            "        const RooCmdArg& arg4=RooCmdArg::none(), const RooCmdArg& arg5=RooCmdArg::none(),const RooCmdArg& arg6=RooCmdArg::none(),"
            "        const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none(),const RooCmdArg& arg9=RooCmdArg::none()) ;",
            "RooChi2Var(const char* name, const char* title, RooAbsPdf& pdf, RooDataHist& data,"
            "        const RooCmdArg& arg1, const RooCmdArg& arg2=RooCmdArg::none(),const RooCmdArg& arg3=RooCmdArg::none(),"
            "        const RooCmdArg& arg4=RooCmdArg::none(), const RooCmdArg& arg5=RooCmdArg::none(),const RooCmdArg& arg6=RooCmdArg::none(),"
            "        const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none(),const RooCmdArg& arg9=RooCmdArg::none()) ;",
        ]
    )
    def __init__(self, *args, **kwargs):
        """The RooCategory constructor is pythonized for converting python dict to std::map.
        The keywords must correspond to the CmdArg of the constructor function.
        """
        # Redefinition of `RooChi2Var` constructor for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        self._init(*args, **kwargs)
