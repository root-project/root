# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

ROOT_ADD_GTEST(testRWLock testRWLock.cxx LIBRARIES Core Thread Hist ${TBB_LIBRARIES})

ROOT_ADD_GTEST(testTThreadedObject testTThreadedObject.cxx LIBRARIES Hist)

if(NOT MSVC OR CMAKE_SIZEOF_VOID_P EQUAL 8)
  ROOT_ADD_GTEST(testInterpreterLock testInterpreterLock.cxx LIBRARIES Imt REPEATS 50)
else()
  ROOT_ADD_GTEST(testInterpreterLock testInterpreterLock.cxx LIBRARIES Imt REPEATS 40)
endif()
