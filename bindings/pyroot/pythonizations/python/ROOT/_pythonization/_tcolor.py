# Author: Vincenzo Eduardo Padulano CERN 11/2024

################################################################################
# Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

import functools

from . import pythonization


def _tcolor_constructor(original_init):
    @functools.wraps(original_init)
    def wrapper(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        import ROOT
        ROOT.SetOwnership(self, False)
    return wrapper


@pythonization("TColor")
def pythonize_tcolor(klass):
    klass.__init__ = _tcolor_constructor(klass.__init__)

