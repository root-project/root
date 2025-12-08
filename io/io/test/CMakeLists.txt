# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

ROOT_ADD_GTEST(RRawFile RRawFile.cxx LIBRARIES RIO)
ROOT_ADD_GTEST(TFile TFileTests.cxx LIBRARIES RIO)
ROOT_ADD_GTEST(TBufferFile TBufferFileTests.cxx LIBRARIES RIO)
ROOT_ADD_GTEST(TBufferMerger TBufferMerger.cxx LIBRARIES RIO Imt Tree)
ROOT_ADD_GTEST(TBufferJSON TBufferJSONTests.cxx LIBRARIES RIO)
ROOT_ADD_GTEST(TFileMerger TFileMergerTests.cxx LIBRARIES RIO Tree Hist)
if(ROOT_NEED_STDCXXFS)
  target_link_libraries(TFileMerger PUBLIC stdc++fs)
endif()
ROOT_ADD_GTEST(TROMemFile TROMemFileTests.cxx LIBRARIES RIO Tree)
if(uring AND NOT DEFINED ENV{ROOTTEST_IGNORE_URING})
  ROOT_ADD_GTEST(RIoUring RIoUring.cxx LIBRARIES RIO)
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(RFileTestIncludes
                              NO_INSTALL_HEADERS
                              HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/RFileTestIncludes.hxx
                              SOURCES RFileTestIncludes.cxx
                              LINKDEF RFileTestIncludesLinkDef.h
                              DEPENDENCIES RIO Tree)
configure_file(RFileTestIncludes.hxx . COPYONLY)
if(MSVC AND NOT CMAKE_GENERATOR MATCHES Ninja)
  add_custom_command(TARGET RFileTestIncludes POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/libRFileTestIncludes.dll
                                     ${CMAKE_CURRENT_BINARY_DIR}/libRFileTestIncludes.dll)
endif()
ROOT_ADD_GTEST(rfile rfile.cxx LIBRARIES RIO Hist ROOTNTuple Tree RFileTestIncludes)
if(pyroot)
  ROOT_ADD_PYUNITTEST(rfile_py rfile.py)
endif()

# Temporarily disabled.  Test routinely fails on MacOS and some Linuxes.
#if(NOT WIN32 AND (NOT MACOS_VERSION OR NOT MACOSX_VERSION VERSION_LESS 13.00))
#  ROOT_EXECUTABLE(TMapFileTest TMapFileTest.cxx LIBRARIES RIO Hist New)
#  ROOT_ADD_TEST(io-io-test-TMapFileTest COMMAND TMapFileTest complete)
#endif()

ROOTTEST_ADD_TEST(testLargeByteCounts
                  MACRO testByteCount.C
                  OUTREF testByteCount.ref)
