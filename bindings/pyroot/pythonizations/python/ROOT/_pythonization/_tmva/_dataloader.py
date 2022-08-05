# Authors:
# * Lorenzo Moneta 06/2022
# * Harshal Shende 06/2022

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################


r"""
/**
\class Dataloader
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


class DataLoader(object):
    @cpp_signature(
        "TMVA::DataLoader::PrepareTrainingAndTestTree( const TCut& cut,"
        "Int_t NsigTrain, Int_t NbkgTrain, Int_t NsigTest, Int_t NbkgTest, const TString& otherOpt )"
    )
    def PrepareTrainingAndTestTree(self, *args, **kwargs):

        r"""Dataloader::PrepareTrainingAndTestTree() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        args, kwargs = _kwargs_to_tmva_cmdargs(*args, **kwargs)
        return self._PrepareTrainingAndTestTree(*args, **kwargs)