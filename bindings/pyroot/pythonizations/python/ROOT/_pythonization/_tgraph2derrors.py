# Author: Harshal Shende CERN  10/2022

################################################################################
# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################


from . import pythonization


def _TGraph2DErrorsConstructor(self, *args):
    import numpy as np

    if isinstance(args[0], (np.ndarray, np.generic)):
        args = (args[0].size,) + args

    self._original__init__(*args)


@pythonization("TGraph2DErrors")
def pythonize_tgraph2derrors(klass):
    # Parameters:
    # klass: class to be pythonized
    # Support hist *= scalar
    klass._original__init__ = klass.__init__
    klass.__init__ = _TGraph2DErrorsConstructor
