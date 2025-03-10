# Author: Vincenzo Eduardo Padulano CERN 11/2024
# Author: Jonas Rembser CERN 11/2024

################################################################################
# Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################
from . import pythonization


def _TFormula_Constructor(self, *args, **kwargs):
    """
    Forward the arguments to the C++ constructor and retain ownership. This
    helps avoiding double deletes due to ROOT automatic memory management.
    """
    self._cpp_constructor(*args, **kwargs)
    import ROOT
    ROOT.SetOwnership(self, False)


@pythonization('TFormula')
def pythonize_tformula(klass):

    klass._cpp_constructor = klass.__init__
    klass.__init__ = _TFormula_Constructor
