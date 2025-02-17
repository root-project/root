# Author: Danilo Piparo, Stefan Wunsch, Massimiliano Galli CERN  08/2018

################################################################################
# Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

r"""
\pythondoc TDirectoryFile

In the same way as for TDirectory, it is possible to inspect the content of a
TDirectoryFile object from Python as if the subdirectories and objects it
contains were its attributes. For more information, please refer to the
TDirectory documentation.

In addition to the attribute syntax, one can inspect a TDirectoryFile in Python
via the `Get` method. In this case, the subdirectory/object name is specified
as a string:
\code{.py}
# Access a subdirectory
d.Get('subdir')

# We can go further down in the hierarchy of directories
d.Get('subdir/subsubdir')

# Access an object (e.g. a histogram) in the directory
d.Get('obj')

# ... or in a subdirectory
d.Get('subdir/obj')

# Wrong attribute: returns null
x = d.Get('wrongAttr')  # x points to null
\endcode

Furthermore, TDirectoryFile inherits a `WriteObject` Python method from
TDirectory. Such method allows to write an object into a TDirectoryFile
with the following syntax:
\code{.py}
# Write object obj with identifier 'keyName'
d.WriteObject(obj, 'keyName')
\endcode

\endpythondoc
"""

from . import pythonization


def _TDirectoryFile_Get(self, namecycle):
    """
    Allow access to objects through the method Get().

    This concerns both TDirectoryFile and TFile, since the latter
    inherits the Get method from the former.
    We decided not to inject this behavior directly in TDirectory
    because this one already has a templated method Get which, when
    invoked from Python, returns an object of the derived class (e.g. TH1F)
    and not a generic TObject.
    In case the object is not found, a null pointer is returned.
    """

    import cppyy

    key = self.GetKey(namecycle)
    if key:
        class_name = key.GetClassName()
        address = self.GetObjectChecked(namecycle, class_name)
        return cppyy.bind_object(address, class_name)
    # no key? for better or worse, call normal Get()
    return self._Get(namecycle)


# Pythonizor function
@pythonization("TDirectoryFile")
def pythonize_tdirectoryfile(klass):
    """
    TDirectoryFile inherits from TDirectory the pythonized attr syntax (__getattr__)
    and WriteObject method.
    On the other side, the Get() method is pythonized only in TDirectoryFile.
    Thus, the situation is now the following:

    1) __getattr__ : TDirectory --> TDirectoryFile --> TFile
        1.1) caches the returned object for future attempts
        1.2) raises AttributeError if object not found

    2) Get() : TDirectoryFile --> TFile
        2.1) does not cache the returned object
        2.2 returns nullptr if object not found
    """

    klass._Get = klass.Get
    klass.Get = _TDirectoryFile_Get
