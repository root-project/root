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

add_dependencies(ReadSpeed RIO Tree TreePlayer ROOTDataFrame)

target_include_directories(ReadSpeed PRIVATE
  ${CMAKE_SOURCE_DIR}/io/io/inc
  ${CMAKE_SOURCE_DIR}/tree/tree/inc
  ${CMAKE_SOURCE_DIR}/tree/treeplayer/inc
  ${CMAKE_SOURCE_DIR}/tree/dataframe/inc
  ${CMAKE_SOURCE_DIR}/core/foundation/inc
  ${CMAKE_SOURCE_DIR}/core/imt/inc
  ${CMAKE_SOURCE_DIR}/math/mathcore/inc
  ${CMAKE_SOURCE_DIR}/math/matrix/inc
  ${CMAKE_SOURCE_DIR}/math/vecops/inc
  ${CMAKE_SOURCE_DIR}/hist/hist/inc
  ${CMAKE_BINARY_DIR}/ginclude
)

ROOT_ADD_TEST_SUBDIRECTORY(test)
