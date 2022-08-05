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


r"""
/**
\class Factory
\brief \parblock \endparblock
\htmlonly
<div class="pyrootbox">
\endhtmlonly
## PyROOT

\htmlonly
</div>
\endhtmlonly
*/
"""


from ._utils import _kwargs_to_tmva_cmdargs, cpp_signature


class Factory(object):
    r"""Some member functions of factory that take a TMVA CmdArg as argument also support keyword arguments.
    This applies to factory::BookMethod().
    """

    @cpp_signature(
        "Factory::Factory(TString jobName, TFile *theTargetFile, TString theOption)"
        ": Configurable(theOption), fTransformations('I'), fVerbose(kFALSE), fVerboseLevel(kINFO), fCorrelations(kFALSE),"
        "fROC(kTRUE), fSilentFile(theTargetFile == nullptr), fJobName(jobName), fAnalysisType(Types::kClassification),"
        "fModelPersistence(kTRUE),"
        "Factory::Factory(TString jobName, TString theOption)"
    )
    def __init__(self, *args, **kwargs):
        # Redefinition of `Factory` constructor for keyword arguments.
        # The keywords must correspond to the CmdArg of the constructor function.
        args, kwargs = _kwargs_to_tmva_cmdargs(*args, **kwargs)
        self._init(*args, **kwargs)

    @cpp_signature(
        "Factory::BookMethod( DataLoader *loader, TString theMethodName, TString methodTitle, TString theOption = "
        " );"
    )
    def BookMethod(self, *args, **kwargs):

        r"""factory::BookMethod() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        args, kwargs = _kwargs_to_tmva_cmdargs(*args, **kwargs)
        return self._BookMethod(*args, **kwargs)