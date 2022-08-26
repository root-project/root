# Authors:
# * Lorenzo Moneta 04/2022
# * Harshal Shende 04/2022

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################


def _kwargs_to_tmva_cmdargs(*args, **kwargs):
    """Helper function to check kwargs with keys that correspond to a function that creates TmvaCmdArg."""

    def getter(k, v):
        # helper function to get CmdArg attribute from `TMVA`
        # Parameters:
        # k: key of the kwarg
        # v: value of the kwarg

        try:
            if isinstance(v, bool):
                return f"{k}" if v else "!" + f"{k}"
            else:
                return f"{k}={v}"
        except:
            raise AttributeError("Unsupported Type passed")

    if kwargs:
        cmdOpt = ""
        first = True
        for k, v in kwargs.items():
            if not first:
                cmdOpt += ":"
            cmdOpt += getter(k, v)
            first = False

        args = args + (cmdOpt,)

    return args, {}


def cpp_signature(sig):
    """Decorator to set the `_cpp_signature` attribute of a function.
    This information can be used to generate the documentation.
    """

    def decorator(func):
        func._cpp_signature = sig
        return func

    return decorator
