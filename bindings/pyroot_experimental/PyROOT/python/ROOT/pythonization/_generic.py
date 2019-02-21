# Author: Stefan Wunsch CERN  06/2018

################################################################################
# Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from libROOTPython import AddPrettyPrintingPyz
from ROOT import pythonization


@pythonization()
def pythonizegeneric(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: string containing the name of the class

    # Add pretty printing via setting the __str__ special function
    AddPrettyPrintingPyz(klass)

    return True
