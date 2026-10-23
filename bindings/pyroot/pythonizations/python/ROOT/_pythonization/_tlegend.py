# Author: Vipul Cariappa 10/2026

################################################################################
# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################
from . import pythonization


def _TLegend_AddEntry(self, *args, **kwargs):
    """
    Allow adding an entry without an object with `legend.AddEntry(0, label)`
    (or `None` / `ROOT.nullptr` in place of 0).

    A Python integer does not convert to a pointer in the overload resolution,
    and a null pointer would be ambiguous between the
    `AddEntry(const TObject*, ...)` and `AddEntry(const char*, ...)`
    overloads, so the object overload is selected explicitly.
    """
    import ROOT

    if args and not kwargs and (args[0] is None or args[0] is ROOT.nullptr or (type(args[0]) is int and args[0] == 0)):
        add_entry = self._AddEntry.__overload__("const TObject*, const char*, Option_t*")
        return add_entry(ROOT.nullptr, *args[1:])

    return self._AddEntry(*args, **kwargs)


@pythonization("TLegend")
def pythonize_tlegend(klass):
    klass._AddEntry = klass.AddEntry
    klass.AddEntry = _TLegend_AddEntry
