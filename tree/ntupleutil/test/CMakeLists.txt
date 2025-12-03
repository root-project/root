# Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# @author Jakob Blomer CERN

ROOT_STANDARD_LIBRARY_PACKAGE(CustomStructUtil
                              NO_INSTALL_HEADERS
                              HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/CustomStructUtil.hxx
                              SOURCES CustomStructUtil.cxx
                              LINKDEF CustomStructUtilLinkDef.h
                              DEPENDENCIES RIO)
configure_file(CustomStructUtil.hxx . COPYONLY)
if(MSVC AND NOT CMAKE_GENERATOR MATCHES Ninja)
  add_custom_command(TARGET CustomStructUtil POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/libCustomStructUtil.dll
                                     ${CMAKE_CURRENT_BINARY_DIR}/libCustomStructUtil.dll)
endif()

ROOT_ADD_GTEST(ntuple_importer ntuple_importer.cxx LIBRARIES ROOTNTupleUtil CustomStructUtil)
ROOT_ADD_GTEST(ntuple_exporter ntuple_exporter.cxx LIBRARIES ROOTNTupleUtil)
ROOT_ADD_GTEST(ntuple_inspector ntuple_inspector.cxx LIBRARIES ROOTNTupleUtil)

if(ROOT_NEED_STDCXXFS)
  target_link_libraries(ntuple_exporter PRIVATE stdc++fs)
endif()
