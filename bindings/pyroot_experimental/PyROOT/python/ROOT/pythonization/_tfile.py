# Author: Danilo Piparo CERN  08/2018

################################################################################
# Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from libROOTPython import PythonizeTFile
from ROOT import pythonization

# Pythonizor function
@pythonization
def pythonize_tfile(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: string containing the name of the class

    if name == 'TFile':
        # C++ pythonizations
        # - tree.branch syntax
        PythonizeTFile(klass)

    return True
