# Author: Vipul Cariappa 10/2026

################################################################################
# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################
from . import pythonization

# Methods of TText that have overloads taking the text as `const char*` and as
# `const wchar_t*`, for which a Python string would be ambiguous: in C++, a
# (narrow) string literal selects the `const char*` overload.
_text_methods = ["DrawText", "DrawTextNDC", "SetText", "PaintText", "PaintTextNDC"]
_text_signature = "Double_t, Double_t, const char*"


def _select_char_overload(cpp_method, name):
    def method(self, *args, **kwargs):
        if len(args) == 3 and not kwargs and isinstance(args[2], str):
            return getattr(self, "_" + name).__overload__(_text_signature)(*args)
        return getattr(self, "_" + name)(*args, **kwargs)

    method.__doc__ = cpp_method.__doc__
    return method


def _TText_constructor(self, *args, **kwargs):
    """
    Construct a TText from a Python string with the `const char*`
    constructor rather than the `const wchar_t*` one, like a string literal
    does in C++.
    """
    if len(args) == 3 and not kwargs and isinstance(args[2], str):
        return type(self)._cpp_constructor.__overload__(_text_signature)(self, *args)
    return self._cpp_constructor(*args, **kwargs)


@pythonization("TText")
def pythonize_ttext(klass):
    klass._cpp_constructor = klass.__init__
    klass.__init__ = _TText_constructor
    for name in _text_methods:
        if hasattr(klass, name):
            cpp_method = getattr(klass, name)
            setattr(klass, "_" + name, cpp_method)
            setattr(klass, name, _select_char_overload(cpp_method, name))
