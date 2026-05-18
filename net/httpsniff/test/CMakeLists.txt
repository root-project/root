# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT net/http package
# @author Sergey Linev, GSI
############################################################################

ROOT_ADD_GTEST(testRootSniffer test_sniffer.cxx LIBRARIES RHTTPSniff)
