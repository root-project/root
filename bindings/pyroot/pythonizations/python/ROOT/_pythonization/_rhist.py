# Author: Vipul Cariappa 10/2026

################################################################################
# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################
from . import pythonization


def _axes_vector(axes):
    """Convert a Python sequence of axes to a std::vector<RAxisVariant>."""
    import ROOT

    vec = ROOT.std.vector["ROOT::Experimental::RAxisVariant"]()
    vec.reserve(len(axes))
    for axis in axes:
        vec.push_back(ROOT.Experimental.RAxisVariant(axis))
    return vec


def _RHist_constructor(self, *args, **kwargs):
    """
    Allow constructing an RHist or RHistEngine from a Python list of axes,
    and from a number of bins and an interval given as a Python tuple.

    In C++, `RHist<int>({axis1, axis2})` uses the constructor taking an
    `std::initializer_list<RAxisVariant>`, but a Python list can't be turned
    into a temporary initializer list of RAxisVariant, which is not default
    constructible. Pass the axes as the equivalent std::vector instead.
    Likewise, a Python tuple is not a braced-init-list for the overload
    resolution, so the interval is converted to a std::pair explicitly.
    """
    import ROOT

    if len(args) == 1 and not kwargs and isinstance(args[0], (list, tuple)):
        args = (_axes_vector(args[0]),)
    elif len(args) == 2 and not kwargs and isinstance(args[1], tuple) and len(args[1]) == 2:
        args = (args[0], ROOT.std.pair["double", "double"](*args[1]))
    self._cpp_constructor(*args, **kwargs)


@pythonization("RHist<", ns="ROOT::Experimental", is_prefix=True)
def pythonize_rhist(klass, name):
    klass._cpp_constructor = klass.__init__
    klass.__init__ = _RHist_constructor


@pythonization("RHistEngine<", ns="ROOT::Experimental", is_prefix=True)
def pythonize_rhistengine(klass, name):
    klass._cpp_constructor = klass.__init__
    klass.__init__ = _RHist_constructor
