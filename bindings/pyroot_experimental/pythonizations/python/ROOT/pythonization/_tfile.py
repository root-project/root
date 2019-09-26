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

from libROOTPythonizations import AddFileOpenPyz
from ROOT import pythonization
from libcppyy import bind_object

def _TFileConstructor(self, *args):
    # Redefinition of ROOT.TFile(str, ...):
    # check if the instance of TFile has IsZombie() = True
    # and raise OSError if so.
    # Parameters:
    # self: instance of TFile class
    # *args: arguments passed to the constructor
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

# Pythonizor function
@pythonization()
def pythonize_tfile(klass, name):

    if name == 'TFile':
        # Pythonizations for TFile::Open
        AddFileOpenPyz(klass)
        klass._OriginalOpen = klass.Open
        klass.Open = classmethod(_TFileOpen)

        # Pythonization for TFile constructor
        klass._OriginalConstructor = klass.__init__
        klass.__init__ = _TFileConstructor

    return True
