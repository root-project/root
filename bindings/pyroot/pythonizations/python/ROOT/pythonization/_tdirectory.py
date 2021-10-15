# Author: Danilo Piparo, Stefan Wunsch CERN  08/2018

################################################################################
# Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

r'''
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
'''

from libROOTPythonizations import AddDirectoryGetAttrPyz, AddDirectoryWritePyz
import cppyy

def pythonize_tdirectory():
    klass = cppyy.gbl.TDirectory
    AddDirectoryGetAttrPyz(klass)
    AddDirectoryWritePyz(klass)

# Instant pythonization (executed at `import ROOT` time), no need of a
# decorator. This is a core class that is instantiated before cppyy's
# pythonization machinery is in place.
pythonize_tdirectory()
