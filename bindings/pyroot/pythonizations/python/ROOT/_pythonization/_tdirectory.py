# Author: Danilo Piparo, Stefan Wunsch CERN  08/2018

################################################################################
# Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

r"""
\pythondoc TDirectory

It is possible to retrieve the content of a TDirectory object
just like getting items from a Python dictionary.
Moreover, once a subdirectory or object is accessed for the first time,
it is cached for later use.
For example, assuming `d` is a TDirectory instance:
\code{.py}
# Access a subdirectory
d["subdir"]

# We can go further down in the hierarchy of directories
d["subdir"]["subsubdir"]

# Access an object (e.g. a histogram) in the directory
d["obj"]

# ... or in a subdirectory
d["subdir"]["obj"]

# Wrong key: raises KeyError
d["wrongAttr"]
\endcode

Furthermore, TDirectory implements a `WriteObject` Python method which relies
on TDirectory::WriteObjectAny. This method is a no-op for TDirectory objects,
but it is useful for objects of TDirectory subclasses such as TDirectoryFile
and TFile, which inherit it. Please refer to the documentation of those classes
for more information.

\endpythondoc
"""

import cppyy


def _TDirectory_getitem(self, key):
    """Injection of TDirectory.__getitem__ that raises AttributeError on failure.

    Method that is assigned to TDirectory.__getitem__. It relies on Get to
    obtain the object from the TDirectory and adds on top:
    - Raising an AttributeError if the object does not exist
    - Caching the result of a successful get for future re-attempts.
    Once cached, the same object is retrieved every time.
    This pythonization is inherited by TDirectoryFile and TFile.

    Example:
    ```
    myfile.mydir.mysubdir.myHist.Draw()
    ```
    """
    if not hasattr(self, "_cached_items"):
        self._cached_items = dict()

    if key in self._cached_items:
        return self._cached_items[key]

    result = self.Get(key)
    if not result:
        raise KeyError(f"{repr(self)} object has no key '{key}'")

    # Caching behavior seems to be more clear to the user; can always override said
    # behavior (i.e. re-read from file) with an explicit Get() call
    self._cached_items[key] = result
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


def _ipython_key_completions_(self):
    r"""
    Support tab completion for `__getitem__`, suggesting the names of all
    objects in the file.
    """
    return [k.GetName() for k in self.GetListOfKeys()]


def pythonize_tdirectory():
    klass = cppyy.gbl.TDirectory
    klass.__getitem__ = _TDirectory_getitem
    klass._WriteObject = klass.WriteObject
    klass.WriteObject = _TDirectory_WriteObject
    klass._ipython_key_completions_ = _ipython_key_completions_


# Instant pythonization (executed at `import ROOT` time), no need of a
# decorator. This is a core class that is instantiated before cppyy's
# pythonization machinery is in place.
pythonize_tdirectory()
