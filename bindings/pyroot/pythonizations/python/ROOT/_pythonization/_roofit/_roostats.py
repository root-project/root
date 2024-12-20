# Authors:
# * Aaron Jomy 08/2024
# * Jonas Rembser 08/2024

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from .. import pythonization

from ._utils import _kwargs_to_roocmdargs, cpp_signature



class SPlot(object):
    r"""The constructor of RooStats::SPlot takes RooCmdArg as arguments also support keyword arguments.
    This also applies to SPlot::AddSWeights. For example, the following code is equivalent in PyROOT:
    \code{.py}
    # Directly passing a RooCmdArg:
    RooStats.SPlot(data, ROOT.RooStats.RooCmdArg("Strategy", 0))

    # With keyword arguments:
    RooStats.SPlot(data, Strategy = 0)
    \endcode"""

    __cpp_name__ = 'RooStats::SPlot'

    @cpp_signature(
        "SPlot::SPlot(const char* name, const char* title, RooDataSet& data, RooAbsPdf* pdf,"
        "        const RooArgList &yieldsList, const RooArgSet &projDeps,"
        "        bool useWeights, bool cloneData, const char* newName,"
        "        const RooCmdArg& arg5, const RooCmdArg& arg6, const RooCmdArg& arg7, const RooCmdArg& arg8):"
        "TNamed(name, title);"
    )
    def _SPlot_init(self, *args, **kwargs):
        r"""The SPlot constructor is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the constructor. The constructor automatically calls AddSWeight() with the given RooCmdArg

        For example, the following code is equivalent in PyROOT:
        \code{.py}
        # Directly passing a RooCmdArg:
        sData = ROOT.RooStats.SPlot("sData", "An SPlot", data, massModel, [zYield, qcdYield], ROOT.RooStats.RooCmdArg("Strategy", 0))

        # With keyword arguments:
        sData = ROOT.RooStats.SPlot("sData", "An SPlot", data, massModel, [zYield, qcdYield], Strategy=0)
        \endcode
        """

        # Pad args with default parameter values, so we can add the extra fit
        # keyword arguments at the very end.
        args = args + ([], True, False, "")[len(args) - 5 :]  # last number is first idx of default arg

        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        self._init(*args, **kwargs)
        
    @cpp_signature(
        "void SPlot::AddSWeight( RooAbsPdf* pdf, const RooArgList &yieldsTmp,"
        "        const RooArgSet &projDeps, bool includeWeights,"
        "        const RooCmdArg& arg5, const RooCmdArg& arg6, const RooCmdArg& arg7, const RooCmdArg& arg8);"
    )
    def _SPlot_AddSWeight(self, *args, **kwargs):
        r"""The SPlot::AddSWeight function is pythonized with the command argument pythonization.
        For example, the following code is equivalent in PyROOT:
        \code{.py}

        splot = ROOT.RooStats.SPlot("sData", "An SPlot", data, massModel, [zYield, qcdYield])
        
        # Directly passing a RooCmdArg:
        splot.AddSWeight(pdf, [zYield, qcdYield], ROOT.RooStats.RooCmdArg("Strategy", 0), ROOT.RooStats.RooCmdArg("InitialHesse", 1))

        # With keyword arguments:
        splot.AddSWeight(pdf, [zYield, qcdYield], Strategy=3, InitialHesse=1)
        \endcode
        """

        # Pad args with default parameter values, so we can add the extra fit
        # keyword arguments at the very end.
        args = args + ([], True)[len(args) - 2 :]  # last number is first idx of default arg

        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        self._AddSWeight(*args, **kwargs)


    @pythonization("SPlot", ns="RooStats")
    def pythonize_splot(klass):
        klass._init = klass.__init__
        klass.__init__ = SPlot._SPlot_init

        klass._AddSWeight = klass.AddSWeight
        klass.AddSWeight = SPlot._SPlot_AddSWeight