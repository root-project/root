# Author: Enric Tejedor CERN  08/2019

################################################################################
# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from ROOT import pythonization

from libROOTPython import AddUsingToClass


@pythonization()
def pythonize_roodatahist(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: string containing the name of the class

    if name == 'RooDataHist':
        # Add 'using' overloads for plotOn from RooAbsData
        AddUsingToClass(klass, 'plotOn')

    return True
