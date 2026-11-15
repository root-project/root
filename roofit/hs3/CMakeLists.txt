# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT rootfit/histfactory package
# @author Carsten Burgard, CERN
############################################################################

add_library(RooFitHS3 SHARED)
target_sources(RooFitHS3
PRIVATE
  src/HS3Document.cxx
  src/HS3Parser.cxx
  src/HS3Upgrade.cxx
  src/HS3Validate.cxx
  src/HS3Writer.cxx
  src/JSONInterface.cxx
  src/JSONIO.cxx
  src/RooJSONFactoryWSTool.cxx
  src/JSONFactories_RooFitCore.cxx
  src/JSONFactories_HistFactory.cxx
  src/JSONIOUtils.cxx
PRIVATE
  FILE_SET private_header_files TYPE HEADERS BASE_DIRS src/ FILES
    src/HS3Builder.h
    src/HS3Document.h
    src/HS3Schema.h
    src/JSONIOUtils.h
    src/JSONReadRecorder.h
    src/static_execute.h
  FILE_SET HEADERS TYPE HEADERS BASE_DIRS inc/ FILES
    inc/RooFit/Detail/JSONInterface.h
    inc/RooFitHS3/JSONIO.h
    inc/RooFitHS3/RooJSONFactoryWSTool.h
  FILE_SET linkDef TYPE HEADERS BASE_DIRS inc/ FILES
    inc/LinkDef.h
)

target_link_libraries(RooFitHS3
PRIVATE
  nlohmann_json::nlohmann_json
PUBLIC
  HistFactory
  RooFit
  RooFitCore
)

# workaround to include privately headers that are not public, from a "friend" target
get_target_property(RooFitCore_Friend_Dirs RooFitCore HEADER_DIRS_friend_header_files)
if(RooFitCore_Friend_Dirs)
  target_include_directories(RooFitHS3 PRIVATE ${RooFitCore_Friend_Dirs})
endif()
get_target_property(RooBatchComputeGeneric_Friend_Dirs RooBatchCompute_GENERIC HEADER_DIRS_friend_header_files)
if(RooBatchComputeGeneric_Friend_Dirs)
  target_include_directories(RooFitHS3 PRIVATE ${RooBatchComputeGeneric_Friend_Dirs})
endif()

ROOT_GENERATE_TARGET_DICTIONARY(RooFitHS3 "-writeEmptyRootPCM")
ROOT_INSTALL_TARGET(RooFitHS3)

ROOT_ADD_TEST_SUBDIRECTORY(test)
