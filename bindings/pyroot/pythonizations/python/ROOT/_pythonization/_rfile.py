# Author: Giacomo Parolini CERN 04/2025

################################################################################
# Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

r'''
\pythondoc RFile

TODO: document RFile

\code{.py}
# TODO code example
\endcode

\endpythondoc
'''

from . import pythonization


class _RFile_Get:
    """
    Allow access to objects through the method Get().
    This is pythonized to allow Get() to be called both with and without a template argument.
    """

    def __init__(self, rfile):
        self._rfile = rfile

    def __call__(self, namecycle):
        """
        Non-templated Get()
        """
        import ROOT
        import cppyy

        key = self._rfile.GetKeyInfo(namecycle)
        if key.has_value():
            key = key.value()
            obj = ROOT.Experimental.Internal.GetRFileObjectFromKey(self._rfile, key)
            return cppyy.bind_object(obj, key.fClassName.c_str())
        # No key
        return None

    def __getitem__(self, template_arg):
        """
        Templated Get()
        """
        def getitem_wrapper(namecycle):
            obj = self._rfile._OriginalGet[template_arg](namecycle)
            return obj if obj else None
        return getitem_wrapper

    
class _RFile_Put:
    """
    Allow writing objects through the method Put().
    This is pythonized to allow Put() to be called both with and without a template argument.
    """

    def __init__(self, rfile):
        self._rfile = rfile

    def __call__(self, namecycle, obj):
        """
        Non-templated Put()
        """
        className = type(obj).__cpp_name__
        self._rfile.Put[className](namecycle, obj)

    def __getitem__(self, template_arg):
        """
        Templated Put()
        """
        return self._rfile._OriginalPut[template_arg]


def _RFileExit(obj, exc_type, exc_val, exc_tb):
    """
    Close the RFile object.
    Signature and return value are imposed by Python, see
    https://docs.python.org/3/library/stdtypes.html#typecontextmanager.
    """
    obj.Close()
    return False


def _RFileOpen(original):
    """
    Pythonization for the factory methods (Recreate, OpenForReading, OpenForUpdate)
    """
    def rfile_open_wrapper(klass, *args):
        rfile = original(*args)
        rfile._OriginalGet = rfile.Get
        rfile.Get = _RFile_Get(rfile)
        rfile._OriginalPut = rfile.Put
        rfile.Put = _RFile_Put(rfile)
        return rfile

    return rfile_open_wrapper


def _RFileInit(rfile):
    """
    Prevent the creation of RFile through constructor (must use a factory method)
    """
    raise NotImplementedError("RFile can only be created via Recreate, OpenForReading or OpenForUpdate")


@pythonization('RFile', ns="ROOT::Experimental")
def pythonize_rfile(klass):
    # Explicitly prevent to create a RFile via ctor
    klass.__init__ = _RFileInit

    # Pythonize factory methods
    klass.OpenForReading = classmethod(_RFileOpen(klass.OpenForReading))
    klass.OpenForUpdate = classmethod(_RFileOpen(klass.OpenForUpdate))
    klass.Recreate = classmethod(_RFileOpen(klass.Recreate))

    # Pythonization for __enter__ and __exit__ methods
    # These make RFile usable in a `with` statement as a context manager
    klass.__enter__ = lambda rfile: rfile
    klass.__exit__ = _RFileExit
