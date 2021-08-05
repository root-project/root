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

from ._utils import _kwargs_to_roocmdargs


class RooWorkspace(object):
    def __init__(self, *args, **kwargs):
        # Redefinition of `RooWorkspace` constructor for keyword arguments.
        # The keywords must correspond to the CmdArg of the constructor function.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        self._init(*args, **kwargs)

    def __getitem__(self, key):
        # To enable accessing objects in the RooWorkspace with dictionary-like syntax.
        # The key is passed to the general `RooWorkspace::obj()` function.
        return self.obj(key)

    def Import(self, *args, **kwargs):
        """
        Support the C++ `import()` as `Import()` in python
        """
        return getattr(self, "import")(*args, **kwargs)


def RooWorkspace_import(self, *args, **kwargs):
    # Redefinition of `RooWorkspace.import()` for keyword arguments.
    # The keywords must correspond to the CmdArg of the `import()` function.
    args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
    return self._import(*args, **kwargs)


setattr(RooWorkspace, "import", RooWorkspace_import)
