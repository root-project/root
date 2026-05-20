# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

ROOT_ADD_GTEST(testTaskArena testRTaskArena.cxx LIBRARIES Imt ${TBB_LIBRARIES} FAILREGEX "" INCLUDE_DIRS ${TBB_INCLUDE_DIRS})
set_target_properties(testTaskArena PROPERTIES COMPILE_FLAGS "${TBB_CXXFLAGS}")
ROOT_ADD_GTEST(testTBBGlobalControl testTBBGlobalControl.cxx LIBRARIES Imt ${TBB_LIBRARIES} INCLUDE_DIRS ${TBB_INCLUDE_DIRS})
set_target_properties(testTBBGlobalControl PROPERTIES COMPILE_FLAGS "${TBB_CXXFLAGS}")
ROOT_ADD_GTEST(testEnableImt testEnableImt.cxx LIBRARIES Imt ${TBB_LIBRARIES} INCLUDE_DIRS ${TBB_INCLUDE_DIRS})
set_target_properties(testEnableImt PROPERTIES COMPILE_FLAGS "${TBB_CXXFLAGS}")
