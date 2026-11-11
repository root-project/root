# Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT readspeed package
# @author Bertrand Bellenot CERN
############################################################################

add_library(ReadSpeed OBJECT)
target_sources(ReadSpeed
PRIVATE
  src/ReadSpeed.cxx
  src/ReadSpeedCLI.cxx
PUBLIC
  FILE_SET HEADERS TYPE HEADERS BASE_DIRS inc/ FILES
    inc/ReadSpeedCLI.hxx
    inc/ReadSpeed.hxx
)
target_sources(ReadSpeed INTERFACE $<TARGET_OBJECTS:ReadSpeed>)
target_link_libraries(ReadSpeed PUBLIC
  RIO
  Tree
  TreePlayer
  Imt
)

ROOT_ADD_TEST_SUBDIRECTORY(test)
