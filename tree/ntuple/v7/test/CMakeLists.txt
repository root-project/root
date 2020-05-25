# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# @author Jakob Blomer CERN

if(NOT dataframe)
  return()
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(CustomStruct
                              NO_INSTALL_HEADERS
                              HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/CustomStruct.hxx
                              SOURCES CustomStruct.cxx
                              LINKDEF CustomStructLinkDef.h
                              DEPENDENCIES RIO)
if(MSVC)
  add_custom_command(TARGET CustomStruct POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/libCustomStruct.dll
                                     ${CMAKE_CURRENT_BINARY_DIR}/libCustomStruct.dll)
endif()

ROOT_ADD_GTEST(ntuple ntuple.cxx LIBRARIES ROOTDataFrame ROOTNTuple MathCore CustomStruct)
ROOT_ADD_GTEST(ntuple_metrics ntuple_metrics.cxx LIBRARIES ROOTDataFrame ROOTNTuple MathCore CustomStruct)
ROOT_ADD_GTEST(ntuple_minifile ntuple_minifile.cxx LIBRARIES ROOTDataFrame ROOTNTuple MathCore CustomStruct)
ROOT_ADD_GTEST(ntuple_packing ntuple_packing.cxx LIBRARIES ROOTDataFrame ROOTNTuple MathCore CustomStruct)
ROOT_ADD_GTEST(ntuple_pages ntuple_pages.cxx LIBRARIES ROOTDataFrame ROOTNTuple MathCore CustomStruct)
ROOT_ADD_GTEST(ntuple_print ntuple_print.cxx LIBRARIES ROOTDataFrame ROOTNTuple MathCore CustomStruct)
ROOT_ADD_GTEST(ntuple_raw ntuple_raw.cxx LIBRARIES ROOTDataFrame ROOTNTuple MathCore CustomStruct)
ROOT_ADD_GTEST(ntuple_show ntuple_show.cxx LIBRARIES ROOTDataFrame ROOTNTuple MathCore CustomStruct)
ROOT_ADD_GTEST(ntuple_types ntuple_types.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(ntuple_zip ntuple_zip.cxx LIBRARIES ROOTDataFrame ROOTNTuple MathCore CustomStruct)
