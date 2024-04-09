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

ROOT_STANDARD_LIBRARY_PACKAGE(Unsplit
                              NO_INSTALL_HEADERS
                              HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/Unsplit.hxx
                              SOURCES Unsplit.cxx
                              LINKDEF UnsplitLinkDef.h
                              DEPENDENCIES RIO)
configure_file(Unsplit.hxx . COPYONLY)
if(MSVC)
  add_custom_command(TARGET Unsplit POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/libUnsplit.dll
                                     ${CMAKE_CURRENT_BINARY_DIR}/libUnsplit.dll)
endif()

if(MSVC)
  set(genreflex ${CMAKE_BINARY_DIR}/bin/genreflex.exe)
  set(command ${CMAKE_COMMAND} -E env "ROOTIGNOREPREFIX=1" ${genreflex})
else()
  set(genreflex ${CMAKE_BINARY_DIR}/bin/genreflex)
  set(command ${CMAKE_COMMAND} -E env "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}"
              "ROOTIGNOREPREFIX=1" ${genreflex})
endif()

configure_file(UnsplitXML.h . COPYONLY)
configure_file(UnsplitSelection.xml . COPYONLY)
add_custom_command(OUTPUT UnsplitXMLDict.cxx UnsplitXMLDict_rdict.pcm
                   COMMAND ${command} UnsplitXML.h -o UnsplitXMLDict.cxx -s UnsplitSelection.xml
                   DEPENDS rootcling ${genreflex} UnsplitXML.h UnsplitSelection.xml)
add_library(UnsplitXML SHARED UnsplitXMLDict.cxx UnsplitXML.cxx)
set_target_properties(UnsplitXML PROPERTIES ${ROOT_LIBRARY_PROPERTIES})
ROOT_ADD_INCLUDE_DIRECTORIES(UnsplitXML)
target_link_libraries(UnsplitXML PUBLIC RIO)
if(MSVC)
  add_custom_command(TARGET UnsplitXML POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/libUnsplitXML.dll
                                     ${CMAKE_CURRENT_BINARY_DIR}/libUnsplitXML.dll)
endif()

ROOT_ADD_GTEST(ntuple_basics ntuple_basics.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(ntuple_bulk ntuple_bulk.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(ntuple_cluster ntuple_cluster.cxx LIBRARIES ROOTNTuple)
ROOT_ADD_GTEST(ntuple_compat ntuple_compat.cxx LIBRARIES ROOTNTuple)
ROOT_ADD_GTEST(ntuple_descriptor ntuple_descriptor.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(ntuple_endian ntuple_endian.cxx LIBRARIES ROOTNTuple)
ROOT_ADD_GTEST(ntuple_friends ntuple_friends.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(ntuple_merger ntuple_merger.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(ntuple_metrics ntuple_metrics.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(ntuple_model ntuple_model.cxx LIBRARIES ROOTNTuple)
ROOT_ADD_GTEST(ntuple_packing ntuple_packing.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(ntuple_pages ntuple_pages.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(ntuple_print ntuple_print.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(ntuple_project ntuple_project.cxx LIBRARIES ROOTNTuple)
ROOT_ADD_GTEST(ntuple_modelext ntuple_modelext.cxx LIBRARIES ROOTNTuple MathCore CustomStruct)
ROOT_ADD_GTEST(ntuple_serialize ntuple_serialize.cxx LIBRARIES ROOTNTuple CustomStruct)
if(NOT MSVC OR llvm13_broken_tests)
  ROOT_ADD_GTEST(ntuple_types ntuple_types.cxx LIBRARIES ROOTNTuple CustomStruct)
  ROOT_GENERATE_DICTIONARY(ProxiedSTLContainerDict ${CMAKE_CURRENT_SOURCE_DIR}/ProxiedSTLContainer.hxx
                         MODULE ntuple_types
                         LINKDEF ProxiedSTLContainerLinkDef.h
                         OPTIONS -inlineInputHeader
                         DEPENDENCIES CustomStruct)
endif()
ROOT_ADD_GTEST(ntuple_view ntuple_view.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(ntuple_zip ntuple_zip.cxx LIBRARIES ROOTNTuple CustomStruct)

ROOT_ADD_GTEST(rfield_check rfield_check.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(rfield_class rfield_class.cxx LIBRARIES ROOTNTuple CustomStruct Physics)
ROOT_ADD_GTEST(rfield_string rfield_string.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(rfield_unsplit rfield_unsplit.cxx LIBRARIES ROOTNTuple Unsplit UnsplitXML)
ROOT_ADD_GTEST(rfield_variant rfield_variant.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(rfield_vector rfield_vector.cxx LIBRARIES ROOTNTuple CustomStruct)

ROOT_ADD_GTEST(ntuple_minifile ntuple_minifile.cxx LIBRARIES ROOTNTuple Tree CustomStruct)
ROOT_ADD_GTEST(ntuple_show ntuple_show.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(ntuple_storage ntuple_storage.cxx LIBRARIES ROOTNTuple MathCore CustomStruct)
ROOT_ADD_GTEST(ntuple_extended ntuple_extended.cxx LIBRARIES ROOTNTuple MathCore CustomStruct)

ROOT_ADD_GTEST(ntuple_parallel_writer ntuple_parallel_writer.cxx LIBRARIES ROOTNTuple)

ROOT_ADD_GTEST(ntuple_limits ntuple_limits.cxx LIBRARIES ROOTNTuple)

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


# RNTuple Python interface tests
if(pyroot)
  ROOT_ADD_PYUNITTEST(ntuple_py_model ntuple_model.py)
endif()
