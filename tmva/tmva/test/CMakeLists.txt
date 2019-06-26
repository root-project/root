# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT TMVA tests.
# @author Kim Albertsson
############################################################################

set(Libraries TMVA)
include_directories(${ROOT_INCLUDE_DIRS})

# Tests using google test
ROOT_ADD_GTEST(TestRandomGenerator
               TestRandomGenerator.cxx
               LIBRARIES ${Libraries})

if(dataframe)
    ROOT_ADD_GTEST(rtensor rtensor.cxx LIBRARIES ROOTVecOps TMVA)
    ROOT_ADD_GTEST(rtensor-iterator rtensor_iterator.cxx LIBRARIES ROOTVecOps TMVA)
    ROOT_ADD_GTEST(rtensor-utils rtensor_utils.cxx LIBRARIES ROOTVecOps TMVA ROOTDataFrame)
endif()

project(tmva-tests)
find_package(ROOT REQUIRED)
