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


class RooAbsPdf(RooAbsReal):
    """Some member functions of RooAbsPdf that take a RooCmdArg as argument also support keyword arguments.
    So far, this applies to RooAbsPdf::fitTo, RooAbsPdf::plotOn, RooAbsPdf::generate, RooAbsPdf::paramOn, RooAbsPdf::createCdf,
    RooAbsPdf::generateBinned, RooAbsPdf::createChi2, RooAbsPdf::prepareMultiGen and RooAbsPdf::createNLL.
    For example, the following code is equivalent in PyROOT:
    \code{.py}
    # Directly passing a RooCmdArg:
    pdf.fitTo(data, ROOT.RooFit.Range("r1"))

    # With keyword arguments:
    pdf.fitTo(data, Range="r1")
    \endcode"""

    @cpp_signature(
        "RooAbsPdf::fitTo(RooAbsData&, const RooCmdArg&, const RooCmdArg&, const RooCmdArg&, const RooCmdArg&, const RooCmdArg&, const RooCmdArg&, const RooCmdArg&, const RooCmdArg&)"
    )
    def fitTo(self, *args, **kwargs):
        """This function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the RooAbsPdf::fitTo() function.
        """
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._fitTo(*args, **kwargs)

    def plotOn(self, *args, **kwargs):
        # Redefinition of `RooAbsPdf.plotOn` for keyword arguments.
        # The keywords must correspond to the CmdArg of the `plotOn` function.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._plotOn(*args, **kwargs)

    def generate(self, *args, **kwargs):
        # Redefinition of `RooAbsPdf.generate` for keyword arguments.
        # The keywords must correspond to the CmdArg of the `generate` function.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._generate(*args, **kwargs)

    def paramOn(self, *args, **kwargs):
        # Redefinition of `RooAbsPdf.paramOn` for keyword arguments.
        # The keywords must correspond to the CmdArg of the `paramOn` function.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._paramOn(*args, **kwargs)

    def createNLL(self, *args, **kwargs):
        # Redefinition of `RooAbsPdf.createNLL` for keyword arguments.
        # The keywords must correspond to the CmdArg of the `createNLL` function.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._createNLL(*args, **kwargs)

    def createChi2(self, *args, **kwargs):
        # Redefinition of `RooAbsPdf.createChi2` for keyword arguments.
        # The keywords must correspond to the CmdArg of the `createChi2` function.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._createChi2(*args, **kwargs)

    def createCdf(self, *args, **kwargs):
        # Redefinition of `RooAbsPdf.createCdf` for keyword arguments.
        # The keywords must correspond to the CmdArg of the `createCdf` function.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._createCdf(*args, **kwargs)

    def generateBinned(self, *args, **kwargs):
        # Redefinition of `RooAbsPdf.generateBinned` for keyword arguments.
        # The keywords must correspond to the CmdArg of the `generateBinned` function.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._generateBinned(*args, **kwargs)

    def prepareMultiGen(self, *args, **kwargs):
        # Redefinition of `RooAbsPdf.prepareMultiGen` for keyword arguments.
        # The keywords must correspond to the CmdArg of the `prepareMultiGen` function.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._prepareMultiGen(*args, **kwargs)
