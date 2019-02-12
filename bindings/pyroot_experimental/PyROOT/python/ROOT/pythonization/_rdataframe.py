# Author: Stefan Wunsch CERN  02/2019

################################################################################
# Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from ROOT import pythonization
from libROOTPython import MakeRDataFrame


# Add MakeRDataFrame feature as free function to the ROOT module
import cppyy
cppyy.gbl.ROOT.RDF.MakeRDataFrame = MakeRDataFrame
