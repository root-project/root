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
\class CrossValidation
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


class CrossValidation(object):
    @cpp_signature(
        "TMVA::CrossValidation::CrossValidation(TString jobName, TMVA::DataLoader *dataloader, TString options)"
    )
    def __init__(self, *args, **kwargs):
        # Redefinition of `CrossValidtion` constructor for keyword arguments.
        # The keywords must correspond to the CmdArg of the constructor function.

        args, kwargs = _kwargs_to_tmva_cmdargs(*args, **kwargs)
        return self._init(*args, **kwargs)