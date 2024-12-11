# Author: Vincenzo Eduardo Padulano 12/2024

################################################################################
# Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

def _should_give_up_ownership(object):
    """
    Ownership of objects which automatically register to a directory should be
    left to C++, except if the object is gROOT.
    """
    import ROOT
    tdir = object.GetDirectory()
    return bool(tdir) and tdir is not ROOT.gROOT

def _constructor_releasing_ownership(self, *args, **kwargs):
    """
    Forward the arguments to the C++ constructor and give up ownership if the
    object is attached to a directory, which is then the owner. The only
    exception is when the owner is gROOT, to avoid introducing a
    backwards-incompatible change.
    """
    import ROOT

    self._cpp_constructor(*args, **kwargs)
    if _should_give_up_ownership(self):
        ROOT.SetOwnership(self, False)

def _SetDirectory_SetOwnership(self, dir):
    self._Original_SetDirectory(dir)
    if dir:
        # If we are actually registering with a directory, give ownership to C++
        import ROOT
        ROOT.SetOwnership(self, False)
