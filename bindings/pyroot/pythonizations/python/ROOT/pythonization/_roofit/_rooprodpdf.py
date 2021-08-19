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


class RooProdPdf(object):
    """RooProdPdf() constructor takes a RooCmdArg as argument also supports keyword arguments.
    For example, the following code is equivalent in PyROOT:
    \code{.py}
    # Directly passing a RooCmdArg:
    model = ROOT.RooProdPdf(
        "model", "model", ROOT.RooArgSet(shapePdf), ROOT.RooFit.Conditional(ROOT.RooArgSet(effPdf), ROOT.RooArgSet(cut))
        )

    # With keyword arguments:
    model = ROOT.RooProdPdf(
        "model", "model", ROOT.RooArgSet(shapePdf), Conditional=(ROOT.RooArgSet(effPdf), ROOT.RooArgSet(cut))
        )
    \endcode
    """

    @cpp_signature(
        "RooProdPdf(const char* name, const char* title, const RooArgSet& fullPdfSet,"
        "    const RooCmdArg& arg1            , const RooCmdArg& arg2=RooCmdArg(),"
        "    const RooCmdArg& arg3=RooCmdArg(), const RooCmdArg& arg4=RooCmdArg(),"
        "    const RooCmdArg& arg5=RooCmdArg(), const RooCmdArg& arg6=RooCmdArg(),"
        "    const RooCmdArg& arg7=RooCmdArg(), const RooCmdArg& arg8=RooCmdArg()) ;"
    )
    def __init__(self, *args, **kwargs):
        """The RooProdPdf constructor is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the constructor.
        """
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        self._init(*args, **kwargs)
