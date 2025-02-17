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
    r"""Constructor of RooChi2Var takes a RooCmdArg as argument also supports keyword arguments."""

    __cpp_name__ = 'RooChi2Var'

    @cpp_signature(
        [
            "RooChi2Var(const char* name, const char* title, RooAbsReal& func, RooDataHist& data,"
            "        const RooCmdArg& arg1, const RooCmdArg& arg2={},const RooCmdArg& arg3={},"
            "        const RooCmdArg& arg4={}, const RooCmdArg& arg5={},const RooCmdArg& arg6={},"
            "        const RooCmdArg& arg7={}, const RooCmdArg& arg8={},const RooCmdArg& arg9={}) ;",
            "RooChi2Var(const char* name, const char* title, RooAbsPdf& pdf, RooDataHist& data,"
            "        const RooCmdArg& arg1, const RooCmdArg& arg2={},const RooCmdArg& arg3={},"
            "        const RooCmdArg& arg4={}, const RooCmdArg& arg5={},const RooCmdArg& arg6={},"
            "        const RooCmdArg& arg7={}, const RooCmdArg& arg8={},const RooCmdArg& arg9={}) ;",
        ]
    )
    def __init__(self, *args, **kwargs):
        r"""The RooCategory constructor is pythonized for converting python dict to std::map.
        The keywords must correspond to the CmdArg of the constructor function.
        """
        # Redefinition of `RooChi2Var` constructor for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        self._init(*args, **kwargs)
