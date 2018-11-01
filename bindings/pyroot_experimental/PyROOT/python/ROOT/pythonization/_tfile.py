# Author: Danilo Piparo CERN  08/2018

################################################################################
# Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from libROOTPython import AddFileOpenPyz
from ROOT import pythonization

# Pythonizor function
@pythonization()
def pythonize_tfile(klass, name):

    if name == 'TFile':
       AddFileOpenPyz(klass)

    return True
