# Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT readspeed package
# @author Bertrand Bellenot CERN
############################################################################

ROOT_OBJECT_LIBRARY(ReadSpeed
  src/ReadSpeed.cxx
  src/ReadSpeedCLI.cxx
)

target_include_directories(ReadSpeed PRIVATE
  ${CMAKE_SOURCE_DIR}/io/io/inc
  ${CMAKE_SOURCE_DIR}/tree/tree/inc
  ${CMAKE_SOURCE_DIR}/tree/treeplayer/inc
  ${CMAKE_SOURCE_DIR}/core/imt/inc
)

ROOT_ADD_TEST_SUBDIRECTORY(test)
