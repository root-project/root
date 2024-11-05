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


class RooNLLVar(object):
    r"""RooNLLVar() constructor takes a RooCmdArg as argument also supports keyword arguments."""

    __cpp_name__ = 'RooNLLVar'

    @cpp_signature(
        "RooNLLVar(const char* name, const char* title, RooAbsPdf& pdf, RooAbsData& data,"
        "    const RooCmdArg& arg1={}, const RooCmdArg& arg2={},const RooCmdArg& arg3={},"
        "    const RooCmdArg& arg4={}, const RooCmdArg& arg5={},const RooCmdArg& arg6={},"
        "    const RooCmdArg& arg7={}, const RooCmdArg& arg8={},const RooCmdArg& arg9={}) ;"
    )
    def __init__(self, *args, **kwargs):
        r"""The RooNLLVar constructor is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArg of the constructor function.
        """
        # Redefinition of `RooNLLVar` constructor for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        self._init(*args, **kwargs)
