# Author: Enric Tejedor CERN  02/2019

################################################################################
# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from . import pythonization
from libROOTPythonizations import AddSetItemTCAPyz


@pythonization('TClonesArray')
def pythonize_tclonesarray(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: string containing the name of the class

    # Add item setter method
    AddSetItemTCAPyz(klass)
