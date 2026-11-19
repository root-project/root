# Author: Massimiliano Galli CERN  06/2019

################################################################################
# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################


import ctypes


def _expand(data, class_name):
    """
    Read back an object pickled by _reduce(): data holds the object as
    streamed by ROOT I/O, class_name is the name of its class.
    """
    import ROOT

    if class_name == "TBufferFile":
        # TBuffer and its derived classes can't stream themselves, but can be
        # created from their contents
        buf = ROOT.TBufferFile(ROOT.TBuffer.kWrite)
        buf.WriteFastArray(data, len(data))
        return buf

    # The buffer is read from in place: keep it alive while it is read
    raw = bytearray(data)
    buf = ROOT.TBufferFile(ROOT.TBuffer.kRead, len(raw), raw, False)
    address = ROOT.addressof(buf.ReadObjectAny(ROOT.nullptr))
    obj = ROOT.bind_object(address, class_name)
    ROOT.SetOwnership(obj, True)
    return obj


def _reduce(obj):
    """
    The __reduce__ method of all C++ instances: stream the object with ROOT
    I/O, to be read back by _expand().
    """
    import ROOT

    class_name = type(obj).__cpp_name__
    if class_name == "TBufferFile":
        # TBuffer and its derived classes can't stream themselves: pickle
        # their contents
        buf = obj
    else:
        if class_name.startswith("__cppjit_internal::Dispatcher"):
            raise OSError(
                "generic streaming of Python objects whose class derives from a C++ class is not supported. "
                "Please refer to the Python pickle documentation for instructions on how to define "
                "a custom __reduce__ method for the derived Python class"
            )
        buf = ROOT.TBufferFile(ROOT.TBuffer.kWrite)
        if buf.WriteObjectAny(obj, ROOT.TClass.GetClass(class_name)) != 1:
            raise OSError("could not stream object of type {}".format(class_name))

    data = ctypes.string_at(ROOT.Internal.GetBufferAddress(buf), buf.Length())
    return _expand, (data, class_name)


def pythonize_cppinstance():
    import cppyy

    cppyy._backend._set_reduce_method(_reduce)


# Instant pythonization (executed at `import ROOT` time), no need of a
# decorator. CPPInstance is the base for cppyy instance proxies and thus needs
# to be always pythonized.
pythonize_cppinstance()
