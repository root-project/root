# Author: Danilo Piparo, Stefan Wunsch CERN  08/2018

################################################################################
# Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

r"""
/**
\class TDirectory
\brief \parblock \endparblock
\htmlonly
<div class="pyrootbox">
\endhtmlonly
## PyROOT

From Python, it is possible to inspect the content of a TDirectory object
as if the subdirectories and objects it contains were its attributes.
Moreover, once a subdirectory or object is accessed for the first time,
it is cached for later use.
For example, assuming `d` is a TDirectory instance:
\code{.py}
# Access a subdirectory
d.subdir

# We can go further down in the hierarchy of directories
d.subdir.subsubdir

# Access an object (e.g. a histogram) in the directory
d.obj

# ... or in a subdirectory
d.subdir.obj

# Wrong attribute: raises AttributeError
d.wrongAttr
\endcode

Furthermore, TDirectory implements a `WriteObject` Python method which relies
on TDirectory::WriteObjectAny. This method is a no-op for TDirectory objects,
but it is useful for objects of TDirectory subclasses such as TDirectoryFile
and TFile, which inherit it. Please refer to the documentation of those classes
for more information.
\htmlonly
</div>
\endhtmlonly
*/
"""

import cppyy


def _TDirectory_getattr(self, attr):
    """Injection of TDirectory.__getattr__ that raises AttributeError on failure.

    Method that is assigned to TDirectory.__getattr__. It relies on Get to
    obtain the object from the TDirectory and adds on top:
    - Raising an AttributeError if the object does not exist
    - Caching the result of a successful get for future re-attempts.
    Once cached, the same object is retrieved every time.
    This pythonisation is inherited by TDirectoryFile and TFile.

    Example:
    ```
    myfile.mydir.mysubdir.myHist.Draw()
    ```
    """
    result = self.Get(attr)
    if not result:
        raise AttributeError(f"{repr(self)} object has no attribute '{attr}'")

    # Caching behavior seems to be more clear to the user; can always override said
    # behavior (i.e. re-read from file) with an explicit Get() call
    setattr(self, attr, result)
    return result


def _TDirectory_WriteObject(self, obj, *args):
    """
    Implements the WriteObject method of TDirectory
    This method allows to write objects into TDirectory instances with this
    syntax:
    ```
    myDir.WriteObject(myObj, "myKeyName")
    ```
    """
    # Implement a check on whether the object is derived from TObject or not.
    # Similarly to what is done in TDirectory::WriteObject with SFINAE.

    if isinstance(obj, cppyy.gbl.TObject):
        return self.WriteTObject(obj, *args)

    return self.WriteObjectAny(obj, type(obj).__cpp_name__, *args)


def pythonize_tdirectory():
    klass = cppyy.gbl.TDirectory
    klass.__getattr__ = _TDirectory_getattr
    klass._WriteObject = klass.WriteObject
    klass.WriteObject = _TDirectory_WriteObject


# Instant pythonization (executed at `import ROOT` time), no need of a
# decorator. This is a core class that is instantiated before cppyy's
# pythonization machinery is in place.
pythonize_tdirectory()
