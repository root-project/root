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

    @cpp_signature(
        "RooPlot *RooSimultaneous::plotOn(RooPlot* frame,"
        "    const RooCmdArg& arg1            , const RooCmdArg& arg2=RooCmdArg(),"
        "    const RooCmdArg& arg3=RooCmdArg(), const RooCmdArg& arg4=RooCmdArg(),"
        "    const RooCmdArg& arg5=RooCmdArg(), const RooCmdArg& arg6=RooCmdArg(),"
        "    const RooCmdArg& arg7=RooCmdArg(), const RooCmdArg& arg8=RooCmdArg(),"
        "    const RooCmdArg& arg9=RooCmdArg(), const RooCmdArg& arg10=RooCmdArg()) const;"
    )
    def plotOn(self, *args, **kwargs):
        r"""The RooSimultaneous::plotOn() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        # Redefinition of `RooSimultaneous.plotOn` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._plotOn(*args, **kwargs)
