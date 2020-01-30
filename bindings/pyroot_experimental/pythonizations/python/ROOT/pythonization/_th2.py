# Author: Enric Tejedor CERN  02/2020

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from ROOT import pythonization

from libROOTPythonizations import AddUsingToClass


@pythonization()
def pythonize_th2(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: name of the class

    if name == 'TH2':
        # Add 'using' overloads from TH1
        AddUsingToClass(klass, 'GetBinErrorUp')
        AddUsingToClass(klass, 'GetBinErrorLow')

    return True
