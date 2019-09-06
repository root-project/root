# Author: Danilo Piparo, Stefan Wunsch CERN  08/2018

################################################################################
# Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from libROOTPython import AddDirectoryGetAttrPyz, AddDirectoryWritePyz
from ROOT import pythonization
import cppyy

# This pythonization must be set as not lazy, otherwise the mechanism cppyy uses
# to pythonize classes will not be able to be triggered on this very core class.
# The pythonization does not have arguments since it is not fired by cppyy but
# manually upon import of the ROOT module.
@pythonization(lazy = False)
def pythonize_tdirectory():
    klass = cppyy.gbl.TDirectory
    AddDirectoryGetAttrPyz(klass)
    AddDirectoryWritePyz(klass)
    return True
