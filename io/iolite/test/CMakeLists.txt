# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT io/iolite tests
############################################################################


ROOT_ADD_GTEST(RRawFile RRawFile.cxx LIBRARIES ROOTIOLite ROOTFoundation)
