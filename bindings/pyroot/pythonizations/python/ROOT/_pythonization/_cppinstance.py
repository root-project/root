# Author: Massimiliano Galli CERN  06/2019

################################################################################
# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

import libcppyy
from libROOTPythonizations import AddCPPInstancePickling

def pythonize_cppinstance():
    klass = libcppyy.CPPInstance

    AddCPPInstancePickling(klass)

# Instant pythonization (executed at `import ROOT` time), no need of a
# decorator. CPPInstance is the base for cppyy instance proxies and thus needs
# to be always pythonized.
pythonize_cppinstance()
