# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# @author Jakob Blomer CERN

ROOT_STANDARD_LIBRARY_PACKAGE(CustomStruct
                              NO_INSTALL_HEADERS
                              HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/CustomStruct.hxx
                              SOURCES CustomStruct.cxx
                              LINKDEF CustomStructLinkDef.h
                              DEPENDENCIES RIO)
configure_file(CustomStruct.hxx . COPYONLY)
if(MSVC AND NOT CMAKE_GENERATOR MATCHES Ninja)
  add_custom_command(TARGET CustomStruct POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/libCustomStruct.dll
                                     ${CMAKE_CURRENT_BINARY_DIR}/libCustomStruct.dll)
endif()

ROOT_ADD_GTEST(ntuple_basics ntuple_basics.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(ntuple_bulk ntuple_bulk.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(ntuple_cast ntuple_cast.cxx LIBRARIES ROOTNTuple)
ROOT_ADD_GTEST(ntuple_checksum ntuple_checksum.cxx ntuple_test.cxx LIBRARIES ROOTNTuple)
ROOT_ADD_GTEST(ntuple_cluster ntuple_cluster.cxx LIBRARIES ROOTNTuple)
ROOT_ADD_GTEST(ntuple_compat ntuple_compat.cxx RXTuple.cxx LIBRARIES ROOTNTuple xxHash::xxHash)
ROOT_GENERATE_DICTIONARY(RXTupleDict ${CMAKE_CURRENT_SOURCE_DIR}/RXTuple.hxx
                       MODULE ntuple_compat
                       LINKDEF RXTupleLinkDef.h
                       DEPENDENCIES RIO)
ROOT_ADD_GTEST(ntuple_descriptor ntuple_descriptor.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_GENERATE_DICTIONARY(RNTupleDescriptorDict ${CMAKE_CURRENT_SOURCE_DIR}/RNTupleDescriptorDict.hxx
                       MODULE ntuple_descriptor
                       LINKDEF RNTupleDescriptorLinkDef.h
                       OPTIONS -inlineInputHeader
                       DEPENDENCIES RIO CustomStruct)
ROOT_ADD_GTEST(ntuple_endian ntuple_endian.cxx LIBRARIES ROOTNTuple)
if(NOT MSVC)
  # These unit tests rely on fork(), which is not available on Windows.
  ROOT_ADD_GTEST(ntuple_evolution ntuple_evolution.cxx LIBRARIES ROOTNTuple)
  ROOT_ADD_GTEST(ntuple_emulated ntuple_emulated.cxx LIBRARIES ROOTNTuple)
endif()
ROOT_ADD_GTEST(ntuple_join_table ntuple_join_table.cxx LIBRARIES ROOTNTuple)
ROOT_ADD_GTEST(ntuple_merger ntuple_merger.cxx LIBRARIES ROOTNTuple CustomStruct ZLIB::ZLIB Tree INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/tree/tree/inc)
ROOT_ADD_GTEST(ntuple_metrics ntuple_metrics.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(ntuple_model ntuple_model.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(ntuple_multi_column ntuple_multi_column.cxx LIBRARIES ROOTNTuple)
ROOT_ADD_GTEST(ntuple_packing ntuple_packing.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(ntuple_pages ntuple_pages.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(ntuple_print ntuple_print.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(ntuple_processor ntuple_processor.cxx LIBRARIES ROOTNTuple)
ROOT_ADD_GTEST(ntuple_processor_chain ntuple_processor_chain.cxx LIBRARIES ROOTNTuple)
ROOT_ADD_GTEST(ntuple_processor_join ntuple_processor_join.cxx LIBRARIES ROOTNTuple)
ROOT_ADD_GTEST(ntuple_project ntuple_project.cxx LIBRARIES ROOTNTuple)
ROOT_ADD_GTEST(ntuple_modelext ntuple_modelext.cxx LIBRARIES ROOTNTuple MathCore CustomStruct)
ROOT_ADD_GTEST(ntuple_serialize ntuple_serialize.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(ntuple_type_name ntuple_type_name.cxx LIBRARIES ROOTNTuple CustomStruct)
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

ROOT_ADD_GTEST(rfield_basics rfield_basics.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(rfield_class rfield_class.cxx LIBRARIES ROOTNTuple CustomStruct Physics)
ROOT_ADD_GTEST(rfield_string rfield_string.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(rfield_variant rfield_variant.cxx LIBRARIES ROOTNTuple CustomStruct)
if(MSVC)
  # Necessary to have TClass handle very large variants
  set_property(TARGET rfield_variant APPEND_STRING PROPERTY LINK_FLAGS " -STACK:8000000")
endif()
ROOT_ADD_GTEST(rfield_vector rfield_vector.cxx LIBRARIES ROOTNTuple CustomStruct)

ROOT_ADD_GTEST(ntuple_minifile ntuple_minifile.cxx LIBRARIES ROOTNTuple Physics Tree CustomStruct)
ROOT_ADD_GTEST(ntuple_show ntuple_show.cxx LIBRARIES ROOTNTuple CustomStruct)
ROOT_ADD_GTEST(ntuple_storage ntuple_storage.cxx LIBRARIES ROOTNTuple MathCore CustomStruct)
ROOT_ADD_GTEST(ntuple_extended ntuple_extended.cxx LIBRARIES ROOTNTuple MathCore CustomStruct)
if(NOT MSVC OR win_broken_tests)
  ROOT_ADD_GTEST(ntuple_largefile1 ntuple_largefile1.cxx LIBRARIES ROOTNTuple MathCore CustomStruct)
  ROOT_ADD_GTEST(ntuple_largefile2 ntuple_largefile2.cxx LIBRARIES ROOTNTuple MathCore CustomStruct)
endif()
ROOT_ADD_GTEST(ntuple_randomaccess ntuple_randomaccess.cxx LIBRARIES ROOTNTuple MathCore CustomStruct)

ROOT_ADD_GTEST(ntuple_parallel_writer ntuple_parallel_writer.cxx LIBRARIES ROOTNTuple)

ROOT_ADD_GTEST(ntuple_limits ntuple_limits.cxx LIBRARIES ROOTNTuple)

ROOT_ADD_GTEST(rfield_streamer rfield_streamer.cxx StreamerField.cxx StreamerFieldXML.cxx LIBRARIES ROOTNTuple)
ROOT_GENERATE_DICTIONARY(StreamerFieldDict ${CMAKE_CURRENT_SOURCE_DIR}/StreamerField.hxx
                         MODULE rfield_streamer LINKDEF StreamerFieldLinkDef.h OPTIONS -inlineInputHeader
                         DEPENDENCIES RIO)

if(MSVC)
  set(genreflex ${CMAKE_BINARY_DIR}/bin/genreflex.exe)
  set(command ${CMAKE_COMMAND} -E env "ROOTIGNOREPREFIX=1" ${genreflex})
else()
  set(genreflex ${CMAKE_BINARY_DIR}/bin/genreflex)
  set(command ${CMAKE_COMMAND} -E env "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}"
              "ROOTIGNOREPREFIX=1" ${genreflex})
endif()

configure_file(StreamerFieldXML.h . COPYONLY)
configure_file(StreamerFieldSelection.xml . COPYONLY)
add_custom_command(OUTPUT StreamerFieldXMLDict.cxx StreamerFieldXMLDict_rdict.pcm
                   COMMAND ${command} StreamerFieldXML.h -o StreamerFieldXMLDict.cxx -s StreamerFieldSelection.xml
                   DEPENDS rootcling StreamerFieldXML.h StreamerFieldSelection.xml)
add_library(StreamerFieldXMLDict OBJECT StreamerFieldXMLDict.cxx)
set_target_properties(StreamerFieldXMLDict PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
target_sources(rfield_streamer PRIVATE $<TARGET_OBJECTS:StreamerFieldXMLDict>)
target_compile_options(StreamerFieldXMLDict PRIVATE $<TARGET_PROPERTY:rfield_streamer,COMPILE_OPTIONS>)
target_compile_definitions(StreamerFieldXMLDict PRIVATE $<TARGET_PROPERTY:rfield_streamer,COMPILE_DEFINITIONS>)
target_compile_features(StreamerFieldXMLDict PRIVATE $<TARGET_PROPERTY:rfield_streamer,COMPILE_FEATURES>)
target_include_directories(StreamerFieldXMLDict PRIVATE $<TARGET_PROPERTY:rfield_streamer,INCLUDE_DIRECTORIES>)
if(MSVC AND NOT CMAKE_GENERATOR MATCHES Ninja)
  add_custom_command(TARGET rfield_streamer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/librfield_streamer_rdict.pcm
                             ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/librfield_streamer_rdict.pcm
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/StreamerFieldXMLDict_rdict.pcm
                             ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/StreamerFieldXMLDict_rdict.pcm)
endif()

if(daos OR daos_mock)
  # Label of the DAOS pool used for testing, if not provided (may be any for libdaos_mock).
  if(NOT daos_test_pool)
    set(daos_test_pool ntuple-daos-test-pool)
  endif()

  ROOT_ADD_GTEST(ntuple_storage_daos ntuple_storage_daos.cxx ntuple_test.cxx LIBRARIES ROOTNTuple MathCore CustomStruct)
  target_compile_definitions(ntuple_storage_daos PRIVATE R__DAOS_TEST_POOL="${daos_test_pool}")

  if(daos_mock)
    set_property(SOURCE ntuple_storage_daos.cxx
            APPEND PROPERTY COMPILE_DEFINITIONS R__DAOS_TEST_MOCK=1)
  endif()
endif()


# RNTuple Python interface tests
if(pyroot)
  ROOT_ADD_PYUNITTEST(ntuple_py_basics ntuple_basics.py)
  ROOT_ADD_PYUNITTEST(ntuple_py_model ntuple_model.py)
endif()
