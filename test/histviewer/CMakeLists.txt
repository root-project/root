# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(HistViewer)

find_package(ROOT REQUIRED)

set(CMAKE_CXX_FLAGS ${ROOT_CXX_FLAGS})

include_directories(${ROOT_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

set(SOURCES
  canvsave.cxx
  histaction.cxx
)

set(HEADERS histaction.h canvsave.h)

ROOT_GENERATE_DICTIONARY(HistViewerDict ${HEADERS} LINKDEF LinkDef.h)
file(COPY ${HEADERS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_library(HistViewer SHARED ${SOURCES} HistViewerDict.cxx)
target_link_libraries(HistViewer ROOT::Core ROOT::Rint ROOT::Gui ROOT::Thread ROOT::RIO ROOT::Hist ROOT::Gpad)
set_target_properties(HistViewer PROPERTIES ENABLE_EXPORTS 1)

install(TARGETS HistViewer RUNTIME DESTINATION ${PROJECT_NAME} COMPONENT applications)
install(FILES ${HEADERS} DESTINATION ${PROJECT_NAME} COMPONENT headers)