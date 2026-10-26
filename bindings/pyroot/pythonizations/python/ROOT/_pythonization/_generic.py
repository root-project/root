# Author: Stefan Wunsch, Enric Tejedor CERN  06/2018

################################################################################
# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################


def _add_getitem_checked(klass):
    # Parameters:
    # - klass: class where to add a __getitem__ method that raises
    # IndexError if index is out of range

    def getitem_checked(o, i):
        # Get item of `o` at `i` or raise IndexError if index is
        # out of range.
        # Assumes `o` has `__len__`.
        # Parameters:
        # - o: object
        # - i: index to be checked in object
        # Returns:
        # - o[i]
        if i >= 0 and i < len(o):
            return o._getitem__unchecked(i)
        else:
            raise IndexError("index out of range")

    klass._getitem__unchecked = klass.__getitem__
    klass.__getitem__ = getitem_checked


def _cling_print_value(self):
    """
    Print the object with the output of cling::printValue, like the ROOT
    prompt does, falling back to __repr__ where that only gives an address.
    """
    import ROOT

    if not ROOT.addressof(self):
        # Null object: cppyy's generic __repr__. The class's own __repr__ may
        # be defined in terms of str(), which would come back here.
        return ROOT._cppyy.types.Instance.__repr__(self)

    result = ROOT.gInterpreter.ToString(type(self).__cpp_name__, self)

    if not result or result.startswith("@0x"):
        # No printer, or cling only gives the address: cppyy's __repr__ says
        # more
        return repr(self)
    return result


# Generic pythonizor for pretty printing that is applied to (almost) all classes
def pythonize_generic(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: string containing the name of the class

    # Add pretty printing via setting the __str__ special function

    # Exclude classes which have the method __str__ already defined in C++
    m = getattr(klass, "__str__", None)
    has_cpp_str = True if m is not None and type(m).__name__ == "CPPOverload" else False

    # Exclude std::string with its own pythonization from cppyy. With the
    # CppInterOp-based backend the class name is the canonical form rather
    # than the "std::string" typedef, so list both shapes.
    exclude = [
        "std::string",
        "std::basic_string<char>",
        "std::basic_string<char,std::char_traits<char>,std::allocator<char> >",
        "std::basic_string<char, std::char_traits<char>, std::allocator<char> >",
    ]

    if name not in exclude and not has_cpp_str:
        klass.__str__ = _cling_print_value
