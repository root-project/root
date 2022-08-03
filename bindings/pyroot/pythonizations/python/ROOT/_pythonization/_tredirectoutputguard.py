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
\class TRedirectOutputGuard
\brief \parblock \endparblock
\htmlonly
<div class="pyrootbox">
\endhtmlonly
## PyROOT

This class can also be used to redirect the C++ stdout and stderr streams within
a Python context manager.

\code{.py}
import ROOT

ROOT.gInterpreter.Declare(\'''
void open_nonexistent_file(){
    TFile f{"nonexistent_file.root", "read"};
}
\''')

with ROOT.TRedirectOutputGuard("mylog.txt", "w"):
    ROOT.open_nonexistent_file()
\endcode

This will save a file named 'mylog.txt' in the current directory, which will
contain the expected output error from ROOT:

\code{.bash}
# mylog.txt
Error in <TFile::TFile>: file $PWD/nonexistent_file.root does not exist
\endcode

\htmlonly
</div>
\endhtmlonly
*/
'''

from . import pythonization


def _TRedirectOutputGuardExit(guard, exc_type, exc_val, exc_tb):
    """
    Destroy the TRedirectOutputGuard, mimicking its RAII behaviour.
    Signature and return value are imposed by Python, see
    https://docs.python.org/3/library/stdtypes.html#typecontextmanager.
    """
    guard.__destruct__()
    return False


@pythonization("TRedirectOutputGuard")
def pythonize_tcontext(klass):
    """
    TRedirectOutputGuard pythonizations:
    - __enter__ and __exit__ methods are defined to make it work as a context manager.
    """

    # The user doesn't need to do anything with the TRedirectOutputGuard object itself, so
    # we don't provide it in the context manager expression
    klass.__enter__ = lambda _: None
    klass.__exit__ = _TRedirectOutputGuardExit
