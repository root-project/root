# Author: Vincenzo Eduardo Padulano CERN 11/2024

################################################################################
# Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################
from . import pythonization


def _TColor_constructor(self, *args, **kwargs):
    """
    Forward the arguments to the C++ constructor and retain ownership. This
    helps avoiding double deletes due to ROOT automatic memory management.
    """
    self._cpp_constructor(*args, **kwargs)
    import ROOT
    ROOT.SetOwnership(self, False)


def _TColor_GetColor(*args, **kwargs):
    """
    Select TColor::GetColor(Float_t r, Float_t g, Float_t b, Float_t a) if
    any of the color components is given as a Python float.

    A Python float is a double, which converts equally well to the Float_t
    and the Int_t components of the two GetColor() overloads, so the call
    would be ambiguous (as it is in C++).
    """
    import ROOT

    get_color = ROOT.TColor._GetColor
    if (
        3 <= len(args) <= 4
        and not kwargs
        and all(isinstance(arg, (int, float)) and not isinstance(arg, bool) for arg in args)
        and any(isinstance(arg, float) for arg in args[:3])
    ):
        return get_color.__overload__("Float_t, Float_t, Float_t, Float_t")(*args)
    return get_color(*args, **kwargs)


@pythonization("TColor")
def pythonize_tcolor(klass):
    klass._cpp_constructor = klass.__init__
    klass.__init__ = _TColor_constructor

    klass._GetColor = klass.GetColor
    klass.GetColor = staticmethod(_TColor_GetColor)
