# Authors:
# * Harshal Shende  03/2021
# * Jonas Rembser 03/2021

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################


from ._rooabsreal import RooAbsReal
from ._utils import _kwargs_to_roocmdargs, cpp_signature


def _pack_cmd_args(*args, **kwargs):
    # Pack command arguments passed into a RooLinkedList.

    import ROOT

    # If the second argument is already a RooLinkedList, do nothing
    if len(kwargs) == 0 and len(args) == 1 and isinstance(args[0], ROOT.RooLinkedList):
        return args[0]

    # Transform keyword arguments to RooCmdArgs
    args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)

    # All keyword arguments should be transformed now
    assert len(kwargs) == 0

    # Put RooCmdArgs in a RooLinkedList
    cmdList = ROOT.RooLinkedList()
    for cmd in args:
        if not isinstance(cmd, ROOT.RooCmdArg):
            raise TypeError("This function only takes RooFit command arguments.")
        cmdList.Add(cmd)

    return cmdList


class RooAbsPdf(RooAbsReal):
    r"""Some member functions of RooAbsPdf that take a RooCmdArg as argument also support keyword arguments.
    So far, this applies to RooAbsPdf::fitTo, RooAbsPdf::plotOn, RooAbsPdf::generate, RooAbsPdf::paramOn, RooAbsPdf::createCdf,
    RooAbsPdf::generateBinned, RooAbsPdf::prepareMultiGen and RooAbsPdf::createNLL.
    For example, the following code is equivalent in PyROOT:
    \code{.py}
    # Directly passing a RooCmdArg:
    pdf.fitTo(data, ROOT.RooFit.Range("r1"))

    # With keyword arguments:
    pdf.fitTo(data, Range="r1")
    \endcode"""

    __cpp_name__ = 'RooAbsPdf'

    @cpp_signature("RooAbsPdf::fitTo()")
    def fitTo(self, *args, **kwargs):
        r"""The RooAbsPdf::fitTo() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        # Redefinition of `RooAbsPdf.fitTo` for keyword arguments.
        return self._fitTo["RooLinkedList const&"](args[0], _pack_cmd_args(*args[1:], **kwargs))

    @cpp_signature(
        "RooPlot *RooAbsPdf::plotOn(RooPlot* frame,"
        "    const RooCmdArg& arg1={}, const RooCmdArg& arg2={},"
        "    const RooCmdArg& arg3={}, const RooCmdArg& arg4={},"
        "    const RooCmdArg& arg5={}, const RooCmdArg& arg6={},"
        "    const RooCmdArg& arg7={}, const RooCmdArg& arg8={},"
        "    const RooCmdArg& arg9={}, const RooCmdArg& arg10={}"
        ") const;"
    )
    def plotOn(self, *args, **kwargs):
        r"""The RooAbsPdf::plotOn() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        # Redefinition of `RooAbsPdf.plotOn` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._plotOn(*args, **kwargs)

    @cpp_signature(
        "RooDataSet *RooAbsPdf::generate(const RooArgSet &whatVars,"
        "    const RooCmdArg& arg1={},const RooCmdArg& arg2={},"
        "    const RooCmdArg& arg3={},const RooCmdArg& arg4={},"
        "    const RooCmdArg& arg5={},const RooCmdArg& arg6={}) ;"
    )
    def generate(self, *args, **kwargs):
        r"""The RooAbsPdf::generate() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        # Redefinition of `RooAbsPdf.generate` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._generate(*args, **kwargs)

    @cpp_signature(
        "RooPlot *RooAbsPdf::paramOn(RooPlot* frame,"
        "    const RooCmdArg& arg1={}, const RooCmdArg& arg2={},"
        "    const RooCmdArg& arg3={}, const RooCmdArg& arg4={},"
        "    const RooCmdArg& arg5={}, const RooCmdArg& arg6={},"
        "    const RooCmdArg& arg7={}, const RooCmdArg& arg8={}) ;"
    )
    def paramOn(self, *args, **kwargs):
        r"""The RooAbsPdf::paramOn() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        # Redefinition of `RooAbsPdf.paramOn` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._paramOn(*args, **kwargs)

    @cpp_signature("RooAbsPdf::createNLL()")
    def createNLL(self, *args, **kwargs):
        r"""The RooAbsPdf::createNLL() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        # Redefinition of `RooAbsPdf.createNLL` for keyword arguments.
        return self._createNLL["RooLinkedList const&"](args[0], _pack_cmd_args(*args[1:], **kwargs))

    @cpp_signature(
        "RooAbsReal *RooAbsPdf::createCdf(const RooArgSet& iset, const RooCmdArg& arg1, const RooCmdArg& arg2={},"
        "    const RooCmdArg& arg3={}, const RooCmdArg& arg4={},"
        "    const RooCmdArg& arg5={}, const RooCmdArg& arg6={},"
        "    const RooCmdArg& arg7={}, const RooCmdArg& arg8={}) ;"
    )
    def createCdf(self, *args, **kwargs):
        r"""The RooAbsPdf::createCdf() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        # Redefinition of `RooAbsPdf.createCdf` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._createCdf(*args, **kwargs)

    @cpp_signature(
        "RooDataHist *RooAbsPdf::generateBinned(const RooArgSet &whatVars,"
        "   const RooCmdArg& arg1={},const RooCmdArg& arg2={},"
        "    const RooCmdArg& arg3={},const RooCmdArg& arg4={},"
        "    const RooCmdArg& arg5={},const RooCmdArg& arg6={}) const;"
    )
    def generateBinned(self, *args, **kwargs):
        r"""The RooAbsPdf::generateBinned() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        # Redefinition of `RooAbsPdf.generateBinned` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._generateBinned(*args, **kwargs)

    @cpp_signature(
        "GenSpec *RooAbsPdf::prepareMultiGen(const RooArgSet &whatVars,"
        "    const RooCmdArg& arg1={},const RooCmdArg& arg2={},"
        "    const RooCmdArg& arg3={},const RooCmdArg& arg4={},"
        "    const RooCmdArg& arg5={},const RooCmdArg& arg6={}) ;"
    )
    def prepareMultiGen(self, *args, **kwargs):
        r"""The RooAbsPdf::prepareMultiGen() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        # Redefinition of `RooAbsPdf.prepareMultiGen` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._prepareMultiGen(*args, **kwargs)
