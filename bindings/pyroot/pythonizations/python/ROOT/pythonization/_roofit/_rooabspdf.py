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
    r"""Some member functions of RooAbsPdf that take a RooCmdArg as argument also support keyword arguments.
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
        r"""The RooAbsPdf::fitTo() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        # Redefinition of `RooAbsPdf.fitTo` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._fitTo(*args, **kwargs)

    @cpp_signature(
        "RooPlot *RooAbsPdf::plotOn(RooPlot* frame,"
        "    const RooCmdArg& arg1=RooCmdArg::none(), const RooCmdArg& arg2=RooCmdArg::none(),"
        "    const RooCmdArg& arg3=RooCmdArg::none(), const RooCmdArg& arg4=RooCmdArg::none(),"
        "    const RooCmdArg& arg5=RooCmdArg::none(), const RooCmdArg& arg6=RooCmdArg::none(),"
        "    const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none(),"
        "    const RooCmdArg& arg9=RooCmdArg::none(), const RooCmdArg& arg10=RooCmdArg::none()"
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
        "    const RooCmdArg& arg1=RooCmdArg::none(),const RooCmdArg& arg2=RooCmdArg::none(),"
        "    const RooCmdArg& arg3=RooCmdArg::none(),const RooCmdArg& arg4=RooCmdArg::none(),"
        "    const RooCmdArg& arg5=RooCmdArg::none(),const RooCmdArg& arg6=RooCmdArg::none()) ;"
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
        "    const RooCmdArg& arg1=RooCmdArg::none(), const RooCmdArg& arg2=RooCmdArg::none(),"
        "    const RooCmdArg& arg3=RooCmdArg::none(), const RooCmdArg& arg4=RooCmdArg::none(),"
        "    const RooCmdArg& arg5=RooCmdArg::none(), const RooCmdArg& arg6=RooCmdArg::none(),"
        "    const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) ;"
    )
    def paramOn(self, *args, **kwargs):
        r"""The RooAbsPdf::paramOn() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        # Redefinition of `RooAbsPdf.paramOn` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._paramOn(*args, **kwargs)

    @cpp_signature(
        "RooAbsReal *RooAbsPdf::createNLL(RooAbsData& data, const RooCmdArg& arg1=RooCmdArg::none(),  const RooCmdArg& arg2=RooCmdArg::none(),"
        "    const RooCmdArg& arg3=RooCmdArg::none(),  const RooCmdArg& arg4=RooCmdArg::none(), const RooCmdArg& arg5=RooCmdArg::none(),"
        "    const RooCmdArg& arg6=RooCmdArg::none(),  const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) ;"
    )
    def createNLL(self, *args, **kwargs):
        r"""The RooAbsPdf::createNLL() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        # Redefinition of `RooAbsPdf.createNLL` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._createNLL(*args, **kwargs)

    @cpp_signature(
        "RooAbsReal *RooAbsPdf::createChi2(RooDataHist& data, const RooCmdArg& arg1=RooCmdArg::none(),  const RooCmdArg& arg2=RooCmdArg::none(),"
        "    const RooCmdArg& arg3=RooCmdArg::none(),  const RooCmdArg& arg4=RooCmdArg::none(), const RooCmdArg& arg5=RooCmdArg::none(),"
        "    const RooCmdArg& arg6=RooCmdArg::none(),  const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) ;"
    )
    def createChi2(self, *args, **kwargs):
        r"""The RooAbsPdf::createChi2() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        # Redefinition of `RooAbsPdf.createChi2` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._createChi2(*args, **kwargs)

    @cpp_signature(
        "RooAbsReal *RooAbsPdf::createCdf(const RooArgSet& iset, const RooCmdArg& arg1, const RooCmdArg& arg2=RooCmdArg::none(),"
        "    const RooCmdArg& arg3=RooCmdArg::none(), const RooCmdArg& arg4=RooCmdArg::none(),"
        "    const RooCmdArg& arg5=RooCmdArg::none(), const RooCmdArg& arg6=RooCmdArg::none(),"
        "    const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) ;"
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
        "   const RooCmdArg& arg1=RooCmdArg::none(),const RooCmdArg& arg2=RooCmdArg::none(),"
        "    const RooCmdArg& arg3=RooCmdArg::none(),const RooCmdArg& arg4=RooCmdArg::none(),"
        "    const RooCmdArg& arg5=RooCmdArg::none(),const RooCmdArg& arg6=RooCmdArg::none()) const;"
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
        "    const RooCmdArg& arg1=RooCmdArg::none(),const RooCmdArg& arg2=RooCmdArg::none(),"
        "    const RooCmdArg& arg3=RooCmdArg::none(),const RooCmdArg& arg4=RooCmdArg::none(),"
        "    const RooCmdArg& arg5=RooCmdArg::none(),const RooCmdArg& arg6=RooCmdArg::none()) ;"
    )
    def prepareMultiGen(self, *args, **kwargs):
        r"""The RooAbsPdf::prepareMultiGen() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        # Redefinition of `RooAbsPdf.prepareMultiGen` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._prepareMultiGen(*args, **kwargs)
