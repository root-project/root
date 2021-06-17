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
\class RooMsgService
\brief \parblock \endparblock
\htmlonly
<div class="pyrootbox">
\endhtmlonly

## PyROOT

Some member functions of RooMsgService that take a RooCmdArg as argument also support keyword arguments.
So far, this applies to RooMsgService::addStream.
For example, the following code is equivalent in PyROOT:
\code{.py}
# Directly passing a RooCmdArg:
ROOT.RooMsgService.instance().addStream(
    ROOT.RooFit.DEBUG, ROOT.RooFit.Topic(ROOT.RooFit.Tracing), ROOT.RooFit.ClassName("RooGaussian")
)

# With keyword arguments:
ROOT.RooMsgService.instance().addStream(ROOT.RooFit.DEBUG, Topic = ROOT.RooFit.Tracing, ClassName = "RooGaussian")
\endcode

\htmlonly
</div>
\endhtmlonly
*/
"""

from ._utils import _kwargs_to_roocmdargs


class RooMsgService(object):
    def addStream(self, *args, **kwargs):
        # Redefinition of `RooMsgService.addStream` for keyword arguments.
        # The keywords must correspond to the CmdArg of the `addStream` function.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._addStream(*args, **kwargs)
