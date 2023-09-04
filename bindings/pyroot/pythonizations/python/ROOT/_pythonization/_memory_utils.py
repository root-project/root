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


def _Clone_releasing_ownership(self, *args, **kwargs):
    """
    Analogous to _constructor_releasing_ownership, but for the TObject::Clone()
    implementation.
    """
    import ROOT

    out = self._Original_Clone(*args, **kwargs)
    if _should_give_up_ownership(out):
        ROOT.SetOwnership(out, False)
    return out


def inject_constructor_releasing_ownership(klass):
    klass._cpp_constructor = klass.__init__
    klass.__init__ = _constructor_releasing_ownership

def inject_clone_releasing_ownership(klass):
    klass._Original_Clone = klass.Clone
    klass.Clone = _Clone_releasing_ownership


def _SetDirectory_SetOwnership(self, dir):
    self._Original_SetDirectory(dir)
    if dir:
        # If we are actually registering with a directory, give ownership to C++
        import ROOT

        ROOT.SetOwnership(self, False)


def declare_cpp_owned_arg(position, name, positional_args, keyword_args, condition=lambda _: True):
    """
    Helper function to drop Python ownership of a specific function argument
    with a given position and name, referring to the C++ signature.

    positional_args and keyword_args should be the usual args and kwargs passed
    to the function, and condition is an optional condition on which the Python
    ownership is dropped.
    """
    import ROOT

    # has to match the C++ argument name
    if name in keyword_args:
        arg = keyword_args[name]
    elif len(positional_args) > position:
        arg = positional_args[0]
    else:
        # This can happen if the function was called with too few arguments.
        # In that case we should not do anything.
        return

    if condition(arg):
        ROOT.SetOwnership(arg, False)
