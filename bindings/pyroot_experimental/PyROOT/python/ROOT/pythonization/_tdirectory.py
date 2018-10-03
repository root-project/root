# Author: Danilo Piparo, Stefan Wunsch CERN  08/2018

################################################################################
# Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from libROOTPython import AddDirectoryAttrSyntaxPyz, AddDirectoryWritePyz
from ROOT import pythonization
import cppyy

# TDirectory::Get method, based on the attr syntax
def _TDirectory_Get(self, objName):
    print "getting " + objName
    return getattr(self, objName)

# This is an instant pythonization. As such, no argument is needed since we know
# what we are pythonizing.

@pythonization(lazy = False)
def pythonize_tdirectory():

    klass = cppyy.gbl.TDirectory
    AddDirectoryAttrSyntaxPyz(klass)
    AddDirectoryWritePyz(klass)
    klass.Get = _TDirectory_Get
    return True
