# Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

include_directories(
  ../res
  ../../clingutils/res
  ../../foundation/res
  ${CLING_INCLUDE_DIRS}
  )

if(root7)
  set(ROOT7SRC TClingDataMemberInfoTestsROOT7.cxx)
endif()

ROOT_ADD_GTEST(
  TClingTest

  TClingTests.cxx
  TClingCallFuncTests.cxx
  TClingDataMemberInfoTests.cxx
  ${ROOT7SRC}
  TClingMethodInfoTests.cxx

  LIBRARIES
  Core
)

add_dependencies(TClingTest Cling RIO)
