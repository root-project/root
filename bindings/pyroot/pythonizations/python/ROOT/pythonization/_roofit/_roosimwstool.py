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
\class RooSimWSTool
\brief \parblock \endparblock
\htmlonly
<div class="pyrootbox">
\endhtmlonly

## PyROOT

Some member functions of RooSimWSTool that take a RooCmdArg as argument also support keyword arguments.
So far, this applies to RooSimWSTool::build.
For example, the following code is equivalent in PyROOT:
\code{.py}
# Directly passing a RooCmdArg:
sct.build("model_sim2", "model", ROOT.RooFit.SplitParam("p0", "c,d"))

# With keyword arguments:
sct.build("model_sim2", "model", SplitParam=("p0", "c,d"))

\endcode

\htmlonly
</div>
\endhtmlonly
*/
"""

from ._utils import _kwargs_to_roocmdargs


class RooSimWSTool(object):
    def build(self, *args, **kwargs):
        # Redefinition of `RooSimWSTool.build` for keyword arguments.
        # The keywords must correspond to the CmdArg of the `build` function.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._build(*args, **kwargs)
