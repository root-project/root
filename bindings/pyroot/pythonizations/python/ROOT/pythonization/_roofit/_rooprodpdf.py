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


r"""
/**
\class RooProdPdf
\brief \parblock \endparblock
\htmlonly
<div class="pyrootbox">
\endhtmlonly

## PyROOT

RooProdPdf() constructor takes a RooCmdArg as argument also supports keyword arguments.
For example, the following code is equivalent in PyROOT:
\code{.py}
# Directly passing a RooCmdArg:
model = ROOT.RooProdPdf(
    "model", "model", ROOT.RooArgSet(shapePdf), ROOT.RooFit.Conditional(ROOT.RooArgSet(effPdf), ROOT.RooArgSet(cut))
)

# With keyword arguments:
model = ROOT.RooProdPdf(
    "model", "model", ROOT.RooArgSet(shapePdf), Conditional=(ROOT.RooArgSet(effPdf), ROOT.RooArgSet(cut))
)
\endcode

\htmlonly
</div>
\endhtmlonly
*/
"""

from ._utils import _kwargs_to_roocmdargs


class RooProdPdf(object):
    def __init__(self, *args, **kwargs):
        # Redefinition of `RooProdPdf` constructor for keyword arguments.
        # The keywords must correspond to the CmdArg of the constructor function.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        self._init(*args, **kwargs)
