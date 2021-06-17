# Author: Stephan Hageboeck, CERN 04/2020

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################


r"""
/**
\class RooWorkspace
\brief \parblock \endparblock
\htmlonly
<div class="pyrootbox">
\endhtmlonly

## PyROOT

The RooWorkspace::import function can't be used in PyROOT because `import` is a reserved python keyword.
For this reason, an alternative with a capitalized name is provided:
\code{.py}

workspace.Import(x)

\endcode

\htmlonly
</div>
\endhtmlonly
*/
"""


class RooWorkspace(object):
    def Import(self, *args, **kwargs):
        """
        Support the C++ `import()` as `Import()` in python
        """
        return getattr(self, "import")(*args, **kwargs)
