# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(RootIDE)

find_package(ROOT REQUIRED)

set(CMAKE_CXX_FLAGS ${ROOT_CXX_FLAGS})

include_directories(${ROOT_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

set(SOURCES
  main.cxx
  TGRootIDE.cxx
)

set(HEADERS TGRootIDE.h)

ROOT_GENERATE_DICTIONARY(RIDEDict ${HEADERS} LINKDEF LinkDef.h)
file(COPY ${HEADERS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_executable(RIDE ${SOURCES} RIDEDict.cxx)
target_link_libraries(RIDE ROOT::Core ROOT::Rint ROOT::Gui ROOT::Thread ROOT::RIO ROOT::Hist ROOT::Gpad)
set_target_properties(RIDE PROPERTIES
  ENABLE_EXPORTS 1
  WINDOWS_EXPORT_ALL_SYMBOLS TRUE
  # The generator expression stops multi-config generators like Visual
  # Studio from appending a per-config subdirectory:
  RUNTIME_OUTPUT_DIRECTORY $<1:${CMAKE_CURRENT_BINARY_DIR}>
)

install(TARGETS RIDE RUNTIME DESTINATION ${PROJECT_NAME} COMPONENT applications)
install(FILES ${HEADERS} DESTINATION ${PROJECT_NAME} COMPONENT headers)
