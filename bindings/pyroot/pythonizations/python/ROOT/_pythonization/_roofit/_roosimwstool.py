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


class RooSimWSTool(object):
    r"""Some member functions of RooSimWSTool that take a RooCmdArg as argument also support keyword arguments.
    So far, this applies to RooSimWSTool::build.
    For example, the following code is equivalent in PyROOT:
    \code{.py}
    # Directly passing a RooCmdArg:
    sct.build("model_sim2", "model", ROOT.RooFit.SplitParam("p0", "c,d"))

    # With keyword arguments:
    sct.build("model_sim2", "model", SplitParam=("p0", "c,d"))
    \endcode
    """

    @cpp_signature(
        "RooSimultaneous *RooSimWSTool::build(const char* simPdfName, const char* protoPdfName,"
        "    const RooCmdArg& arg1=RooCmdArg::none(),const RooCmdArg& arg2=RooCmdArg::none(),"
        "    const RooCmdArg& arg3=RooCmdArg::none(),const RooCmdArg& arg4=RooCmdArg::none(),"
        "    const RooCmdArg& arg5=RooCmdArg::none(),const RooCmdArg& arg6=RooCmdArg::none()) ;"
    )
    def build(self, *args, **kwargs):
        r"""The RooSimWSTool::build() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        # Redefinition of `RooSimWSTool.build` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._build(*args, **kwargs)
