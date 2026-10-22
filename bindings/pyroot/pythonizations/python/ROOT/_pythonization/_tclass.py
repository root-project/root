# Author: Enric Tejedor CERN  02/2019

################################################################################
# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################


def _TClass_DynamicCast(self, base, obj, up=True):
    """
    TClass::DynamicCast returns a void* that the user still has to cast (it
    will have the proper offset, though). Fix this by returning a proxy of the
    target class: `base` for an upcast, this class for a downcast.

    `obj` can also be given as an integer address.
    """
    import cppyy

    if isinstance(obj, int):
        obj = cppyy.ll.cast["void*"](obj)

    address = cppyy.addressof(self._TClass__DynamicCast(base, obj, up))
    return cppyy.bind_object(address, (base if up else self).GetName())


def pythonize_tclass():
    import ROOT

    klass = ROOT.TClass

    # DynamicCast
    klass._TClass__DynamicCast = klass.DynamicCast
    klass.DynamicCast = _TClass_DynamicCast


# Instant pythonization (executed at `import ROOT` time), no need of a
# decorator. This is a core class that is instantiated before cppyy's
# pythonization machinery is in place.
pythonize_tclass()
