# Author: Danilo Piparo, Massimiliano Galli CERN  08/2018

################################################################################
# Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

# TFile inherits from
# - TDirectory the pythonized attr syntax (__getattr__) and WriteObject method.
# - TDirectoryFile the pythonized Get method (pythonized only in Python)
# what is left to add is the pythonization of TFile::Open.

from libROOTPython import AddFileOpenPyz
from ROOT import pythonization
import sys, errno, os
from functools import partial

# We check the Python version because FileNotFoundError
# is not available in Python2
def CheckExistanceAndRaiseError(file_name):
    from cppyy.gbl import gSystem
    if isinstance(file_name, str) and gSystem.AccessPathName(file_name):
        if sys.version_info > (3,0):
            raise FileNotFoundError(errno.ENOENT,
                    os.strerror(errno.ENOENT),
                    file_name)
        else:
            raise IOError(errno.ENOENT,
                    os.strerror(errno.ENOENT),
                    file_name)

def _TFileConstructor(self, *args):
    # Redefinition of ROOT.TFile(str, ...):
    # checks if a file with the given name exists and raises
    # a FileNotFoundError-like if not.
    # After the check, the method is executed anyway and the
    # value returned even if the file does not exist; this is done
    # in order to be consistent with the C++ behavior.
    # Parameters:
    # self: instance of TFile class
    # *args: arguments passed to the constructor; only args[0]
    # (string containing the name of the file) is mandatory

    # Check the existance of the file (and fail) only in READ
    # mode
    if len(args) == 1 or (len(args) == 2 and args[1] == 'READ'):
        CheckExistanceAndRaiseError(args[0])
    tfile_instance = self._OriginalConstructor(*args)
    return tfile_instance

def _TFileOpen(klass, *args):
    # Redefinition of ROOT.TFile.Open(str, ...):
    # checks if a file with the given name exists and raises
    # a FileNotFoundError-like if not.
    # After the check, the method is executed anyway and the
    # value returned even if the file does not exist; this is done
    # in order to be consistent with the C++ behavior.
    # Parameters:
    # klass: TFile class
    # *args: arguments passed to the constructor; only args[0]
    # (string containing the name of the file) is mandatory

    # Check the existance of the file (and fail) only in READ
    # mode
    if len(args) == 1 or (len(args) == 2 and args[1] == 'READ'):
        CheckExistanceAndRaiseError(args[0])
    tfile_instance = klass._OriginalOpen(*args)
    return tfile_instance

# Pythonizor function
@pythonization()
def pythonize_tfile(klass, name):

    if name == 'TFile':
        AddFileOpenPyz(klass)

        klass._OriginalConstructor = klass.__init__
        klass.__init__ = _TFileConstructor
        klass._OriginalOpen = klass.Open
        klass.Open = partial(_TFileOpen, klass)

    return True
