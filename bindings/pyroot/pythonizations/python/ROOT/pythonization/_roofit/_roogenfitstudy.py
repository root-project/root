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
\class RooGenFitStudy
\brief \parblock \endparblock
\htmlonly
<div class="pyrootbox">
\endhtmlonly

## PyROOT

Some member functions of RooGenFitStudy that take a RooCmdArg as argument also support keyword arguments.
So far, this applies to RooGenFitStudy::setGenConfig.

\htmlonly
</div>
\endhtmlonly
*/
"""

from ._utils import _kwargs_to_roocmdargs


class RooGenFitStudy(object):
    def setGenConfig(self, *args, **kwargs):
        # Redefinition of `RooGenFitStudy.setGenConfig` for keyword arguments.
        # The keywords must correspond to the CmdArg of the `setGenConfig` function.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._setGenConfig(*args, **kwargs)
