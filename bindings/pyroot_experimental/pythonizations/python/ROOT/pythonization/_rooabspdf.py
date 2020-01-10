# Author: Stephan Hagebock CERN 01/2020

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
def pythonize_rooabspdf(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: string containing the name of the class

    if name == 'RooAbsPdf':
        # Add 'using' overloads for stuff that comes from RooAbsArg
        AddUsingToClass(klass, 'createChi2')
        AddUsingToClass(klass, 'chi2FitTo')

    return True
