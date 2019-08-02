# Author: Danilo Piparo, Stefan Wunsch, Massimiliano Galli CERN  08/2018

################################################################################
# Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from libROOTPython import AddTDirectoryFileGetPyz
from ROOT import pythonization

"""
TDirectoryFile inherits from TDirectory the pythonized attr syntax (__getattr__)
and WriteObject method.
On the other side, the Get() method is pythonised only in TDirectoryFile.
Thus, the situation is now the following:

    1) __getattr__ : TDirectory --> TDirectoryFile --> TFile
        1.1) caches the returned object for future attempts
        1.2) raises AttributeError if object not found

    2) Get() : TDirectoryFile --> TFile
        2.1) does not cache the returned object
        2.2 returns nullptr if object not found

"""

# Pythonizor function
@pythonization()
def pythonize_tdirectoryfile(klass, name):

    if name == 'TDirectoryFile':
        AddTDirectoryFileGetPyz(klass)

    return True
