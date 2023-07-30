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
configure_file(CustomStruct.hxx . COPYONLY)
if(MSVC)
  add_custom_command(TARGET CustomStruct POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/libCustomStruct.dll
                                     ${CMAKE_CURRENT_BINARY_DIR}/libCustomStruct.dll)
endif()

ROOT_ADD_GTEST(ntuple_basics ntuple_basics.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(ntuple_bulk ntuple_bulk.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(ntuple_cluster ntuple_cluster.cxx LIBRARIES ROOTNTuple)
ROOT_ADD_GTEST(ntuple_descriptor ntuple_descriptor.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(ntuple_endian ntuple_endian.cxx LIBRARIES ROOTNTuple)
ROOT_ADD_GTEST(ntuple_friends ntuple_friends.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(ntuple_merger ntuple_merger.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(ntuple_metrics ntuple_metrics.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(ntuple_packing ntuple_packing.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(ntuple_pages ntuple_pages.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(ntuple_print ntuple_print.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(ntuple_project ntuple_project.cxx LIBRARIES ROOTNTuple)
ROOT_ADD_GTEST(ntuple_rdf ntuple_rdf.cxx LIBRARIES ROOTDataFrame ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(ntuple_modelext ntuple_modelext.cxx LIBRARIES ROOTNTuple MathCore CustomStruct)
ROOT_ADD_GTEST(ntuple_serialize ntuple_serialize.cxx LIBRARIES ROOTNTuple CustomStruct)
if(NOT MSVC OR llvm13_broken_tests)
  ROOT_ADD_GTEST(ntuple_types ntuple_types.cxx LIBRARIES ROOTNTuple CustomStruct)
endif()
ROOT_ADD_GTEST(ntuple_view ntuple_view.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(ntuple_zip ntuple_zip.cxx LIBRARIES ROOTNTuple CustomStruct)

ROOT_ADD_GTEST(rfield_class rfield_class.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(rfield_string rfield_string.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(rfield_variant rfield_variant.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(rfield_vector rfield_vector.cxx LIBRARIES ROOTNTuple CustomStruct)

ROOT_ADD_GTEST(ntuple_minifile ntuple_minifile.cxx LIBRARIES ROOTNTuple Tree CustomStruct)
ROOT_ADD_GTEST(ntuple_show ntuple_show.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(ntuple_storage ntuple_storage.cxx LIBRARIES ROOTNTuple MathCore CustomStruct)
ROOT_ADD_GTEST(ntuple_extended ntuple_extended.cxx LIBRARIES ROOTNTuple MathCore CustomStruct)

if(daos OR daos_mock)
  # Label of the DAOS pool used for testing, if not provided (may be any for libdaos_mock).
  if(NOT daos_test_pool)
    set(daos_test_pool ntuple-daos-test-pool)
  endif()

  ROOT_ADD_GTEST(ntuple_storage_daos ntuple_storage_daos.cxx LIBRARIES ROOTNTuple MathCore CustomStruct)
  target_compile_definitions(ntuple_storage_daos PRIVATE R__DAOS_TEST_POOL="${daos_test_pool}")

  if(daos_mock)
    set_property(SOURCE ntuple_storage_daos.cxx
            APPEND PROPERTY COMPILE_DEFINITIONS R__DAOS_TEST_MOCK=1)
  endif()
endif()
