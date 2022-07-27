# Author: Danilo Piparo, Massimiliano Galli CERN  08/2018
# Author: Vincenzo Eduardo Padulano CERN/UPV 03/2022

################################################################################
# Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

r'''
/**
\class TFile
\brief \parblock \endparblock
\htmlonly
<div class="pyrootbox">
\endhtmlonly
## PyROOT

In the same way as for TDirectory, it is possible to inspect the content of a
TFile object from Python as if the directories and objects it contains were its
attributes. For more information, please refer to the TDirectory documentation.

In addition, TFile instances can be inspected via the `Get` method, a feature
that is inherited from TDirectoryFile (please see the documentation of
TDirectoryFile for examples on how to use it).

In order to write objects into a TFile, the `WriteObject` Python method can
be used (more information in the documentation of TDirectoryFile).

PyROOT modifies the TFile constructor and the TFile::Open method to make them
behave in a more pythonic way. In particular, they both throw an `OSError` if
there was a problem accessing the file (e.g. non-existent or corrupted file).

This class can also be used as a context manager, with the goal of opening a
file and doing some quick manipulations of the objects inside it. The
TFile::Close method will be automatically called at the end of the context. For
example:
\code{.py}
from ROOT import TFile
with TFile("file1.root", "recreate") as outfile:
    hout = ROOT.TH1F(...)
    outfile.WriteObject(hout, "myhisto")
\endcode

Since the file is closed at the end of the context, all objects created or read
from the file inside the context are not accessible anymore in the application
(but they will be stored in the file if they were written to it). ROOT objects
like histograms can be detached from a file with the SetDirectory method. This
will leave the object untouched so that it can be accessed after the end of the
context:
\code{.py}
import ROOT
from ROOT import TFile
with TFile("file1.root", "read") as infile:
    hin = infile.Get("myhisto")
    hin.SetDirectory(ROOT.nullptr)

# Use the histogram afterwards
print(hin.GetName())
\endcode

\note The TFile::Close method automatically sets the current directory in
the program to the gROOT object. If you want to restore the status of the
current directory to some other file that was opened prior to the `with`
statement, you can use the context manager functionality offered by TContext.

\htmlonly
</div>
\endhtmlonly
*/
'''

from libROOTPythonizations import AddFileOpenPyz
from . import pythonization
from libcppyy import bind_object


def _TFileConstructor(self, *args):
    # Redefinition of ROOT.TFile(str, ...):
    # check if the instance of TFile has IsZombie() = True
    # and raise OSError if so.
    # Parameters:
    # self: instance of TFile class
    # *args: arguments passed to the constructor
    if len(args) > 0 and "://" in args[0]:
        raise ValueError("Cannot handle path to remote file '{}' in TFile constructor. Use TFile::Open instead.".format(args[0]))
    self._OriginalConstructor(*args)
    if len(args) >= 1:
        if self.IsZombie():
            raise OSError('Failed to open file {}'.format(args[0]))


def _TFileOpen(klass, *args):
    # Redefinition of ROOT.TFile.Open(str, ...):
    # check if the instance of TFile is a C++ nullptr and raise a
    # OSError if this is the case.
    # Parameters:
    # klass: TFile class
    # *args: arguments passed to the constructor
    f = klass._OriginalOpen(*args)
    if f == bind_object(0, klass):
        # args[0] can be either a string or a TFileOpenHandle
        raise OSError('Failed to open file {}'.format(str(args[0])))
    return f


def _TFileExit(obj, exc_type, exc_val, exc_tb):
    """
    Close the TFile object.
    Signature and return value are imposed by Python, see
    https://docs.python.org/3/library/stdtypes.html#typecontextmanager.
    """
    obj.Close()
    return False


@pythonization('TFile')
def pythonize_tfile(klass):
    """
    TFile inherits from
    - TDirectory the pythonized attr syntax (__getattr__) and WriteObject method.
    - TDirectoryFile the pythonized Get method (pythonized only in Python)
    and defines the __enter__ and __exit__ methods to work as a context manager.
    """

    # Pythonizations for TFile::Open
    AddFileOpenPyz(klass)
    klass._OriginalOpen = klass.Open
    klass.Open = classmethod(_TFileOpen)

    # Pythonization for TFile constructor
    klass._OriginalConstructor = klass.__init__
    klass.__init__ = _TFileConstructor

    # Pythonization for __enter__ and __exit__ methods
    # These make TFile usable in a `with` statement as a context manager
    klass.__enter__ = lambda tfile: tfile
    klass.__exit__ = _TFileExit
