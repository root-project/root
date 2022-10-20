# Authors:
# * Lorenzo Moneta 10/2022
# * Harshal Shende 10/2022

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################


r"""
/**
\class Envelope
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


class Envelope(object):
    r"""Some member functions of envelope that take a TMVA CmdArg as argument also support keyword arguments.
    This applies to Envelope::BookMethod().
    """

    @cpp_signature(
        " TMVA::Envelope::Envelope(const TString &name, DataLoader *dataloader = nullptr, TFile *file = nullptr, const TString options = ""):"
    )
    def __init__(self, *args, **kwargs):
        # Redefinition of `Envelope` constructor for keyword arguments.
        # The keywords must correspond to the CmdArg of the constructor function.
        args, kwargs = _kwargs_to_tmva_cmdargs(*args, **kwargs)
        self._init(*args, **kwargs)

    @cpp_signature(
        "TMVA::Envelope::BookMethod	(TString methodName, TString methodTitle, TString options = "");"
        "TMVA::Envelope::BookMethod	(Types::EMVA method, TString methodTitle, TString options = "");"
    )
    def BookMethod(self, *args, **kwargs):

        r"""BookMethod() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        args, kwargs = _kwargs_to_tmva_cmdargs(*args, **kwargs)
        return self._BookMethod(*args, **kwargs)