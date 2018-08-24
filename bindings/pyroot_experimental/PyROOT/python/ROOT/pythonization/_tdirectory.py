# Author: Danilo Piparo, Stefan Wunsch CERN  08/2018

################################################################################
# Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from libROOTPython import PythonizeTDirectory
from ROOT import instant_pythonization
import cppyy

# This is an instant pythonization. As such, no argument is needed since we know
# what we are pythonizing.

@instant_pythonization
def pythonize_tdirectory():

    PythonizeTDirectory(cppyy.gbl.TDirectory)

    return True
