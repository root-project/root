# Author: Vincenzo Eduardo Padulano CERN/UPV 2022

################################################################################
# Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

r'''
/**
\class TDirectory::TContext
\brief \parblock \endparblock
\htmlonly
<div class="pyrootbox">
\endhtmlonly
## PyROOT

The functionality offered by TContext can be used in PyROOT with a context manager. Here are a few examples:
\code{.py}
import ROOT
from ROOT import TDirectory

with TDirectory.TContext():
    # Open some file here
    file = ROOT.TFile(...)
    # Retrieve contents from the file
    histo = file.Get("myhisto")

# After the 'with' statement, the current directory is restored to ROOT.gROOT
\endcode
\n 
\code{.py}
import ROOT
from ROOT import TDirectory

file1 = ROOT.TFile("file1.root", "recreate")
#...
file2 = ROOT.TFile("file2.root", "recreate")
#...
file3 = ROOT.TFile("file3.root", "recreate")

# Before the 'with' statement, the current directory is file3 (the last file opened)
with TDirectory.TContext(file1):
    # Inside the statement, the current directory is file1
    histo = ROOT.TH1F(...)
    histo.Write()

# After the statement, the current directory is restored to file3
\endcode
\n 
\code{.py}
import ROOT
from ROOT import TDirectory

file1 = ROOT.TFile("file1.root")
file2 = ROOT.TFile("file2.root")

with TDirectory.TContext(file1, file2):
    # Manage content in file2
    histo = ROOT.TH1F(...)
    # histo will be written to file2
    histo.Write()

# Current directory is set to 'file1.root'
\endcode

Note that TContext restores the current directory to its status before the 'with'
statement, but does not change the status of any file that has been opened inside
the context (e.g. it does not automatically close the file).
\htmlonly
</div>
\endhtmlonly
*/
'''

from . import pythonization


def _TContextExit(ctxt, exc_type, exc_val, exc_tb):
    """
    Destroy the TContext, mimicking its RAII behaviour.
    Signature and return value are imposed by Python, see
    https://docs.python.org/3/library/stdtypes.html#typecontextmanager.
    """
    ctxt.__destruct__()
    return False


@pythonization("TContext", ns="TDirectory")
def pythonize_tcontext(klass):
    """
    TContext pythonizations:
    - __enter__ and __exit__ methods are defined to make it work as a context manager.
    """

    # The user doesn't need to do anything with the TContext object itself, so
    # we don't provide it in the context manager expression
    klass.__enter__ = lambda ctxt: None
    klass.__exit__ = _TContextExit
