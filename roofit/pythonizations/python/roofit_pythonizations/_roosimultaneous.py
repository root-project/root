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


from ._utils import _kwargs_to_roocmdargs, cpp_signature, _dict_to_flat_map


class RooSimultaneous(object):
    r"""Some member functions of RooSimultaneous that take a RooCmdArg as argument also support keyword arguments.
    So far, this applies to RooSimultaneous::plotOn.
    For example, the following code is equivalent in PyROOT:
    \code{.py}
    # Directly passing a RooCmdArg:
    pdfSim.plotOn(frame, ROOT.RooFit.Slice(sample,"control"), ROOT.RooFit.ProjWData(sampleSet, combData))

    # With keyword arguments:
    simPdf.plotOn(frame, Slice=(sample, "control"), ProjWData=(sampleSet, combData))
    \endcode
    """

    __cpp_name__ = 'RooSimultaneous'

    @cpp_signature(
        "RooSimultaneous(const char *name, const char *title,"
        "                std::map<std::string,RooAbsPdf*> pdfMap, RooAbsCategoryLValue& inIndexCat) ;"
    )
    def __init__(self, *args):
        r"""The RooSimultaneous constructor that takes a map of category names
        to PDFs is accepting a Python dictionary in Python.
        """
        if len(args) >= 3 and isinstance(args[2], dict):
            args = list(args)
            args[2] = _dict_to_flat_map(args[2], {"std::string": "RooAbsPdf*"})
        self._init(*args)

    @cpp_signature(
        "RooPlot *RooSimultaneous::plotOn(RooPlot* frame,"
        "    const RooCmdArg& arg1            , const RooCmdArg& arg2={},"
        "    const RooCmdArg& arg3={}, const RooCmdArg& arg4={},"
        "    const RooCmdArg& arg5={}, const RooCmdArg& arg6={},"
        "    const RooCmdArg& arg7={}, const RooCmdArg& arg8={},"
        "    const RooCmdArg& arg9={}, const RooCmdArg& arg10={}) const;"
    )
    def plotOn(self, *args, **kwargs):
        r"""The RooSimultaneous::plotOn() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        # Redefinition of `RooSimultaneous.plotOn` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._plotOn(*args, **kwargs)
