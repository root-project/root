# Author: PyROOT Team

################################################################################
# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

r"""
\pythondoc TROOT

Pythonizations for TROOT class, enabling dynamic batch mode toggling.
"""

from . import pythonization


def _TROOT_SetBatch(self, batch=True):
    """
    Sets the batch mode on ROOT.
    If batch mode is turned off in the middle of a session, ensure the
    interactive graphics environment is initialized.
    """
    import ROOT

    self._cpp_SetBatch(batch)
    if not batch and hasattr(ROOT, "app") and ROOT.PyConfig.StartGUIThread:
        ROOT.app.init_graphics(ROOT._cppyy.gbl.gEnv, ROOT._cppyy.gbl.gSystem)


@pythonization("TROOT")
def pythonize_troot(klass):
    klass._cpp_SetBatch = klass.SetBatch
    klass.SetBatch = _TROOT_SetBatch
