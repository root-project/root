# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT rootfit/histfactory package
# @author Pere Mato, CERN
############################################################################

add_library(RooFitCodegen SHARED)
target_sources(RooFitCodegen
PRIVATE
  src/CodegenImpl.cxx
PUBLIC
  FILE_SET HEADERS TYPE HEADERS BASE_DIRS inc/ FILES
    inc/RooFit/CodegenImpl.h
)
target_link_libraries(RooFitCodegen PUBLIC
  RooFit
  RooFitCore
  HistFactory
)
ROOT_GENERATE_TARGET_DICTIONARY(RooFitCodegen "-writeEmptyRootPCM")
ROOT_INSTALL_TARGET(RooFitCodegen)

# workaround to include privately headers that are not public, from a "friend" target
get_target_property(RooFitCore_Friend_Dirs RooFitCore HEADER_DIRS_friend_header_files)
if(RooFitCore_Friend_Dirs)
  target_include_directories(RooFitCodegen PRIVATE ${RooFitCore_Friend_Dirs})
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
