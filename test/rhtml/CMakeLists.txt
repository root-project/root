# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(rhtml)

find_package(ROOT REQUIRED)

set(CMAKE_CXX_FLAGS ${ROOT_CXX_FLAGS})

include_directories(${ROOT_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

set(SOURCES
  rhtml.cxx
)

set(HEADERS rhtml.h)

ROOT_GENERATE_DICTIONARY(rhtmlDict ${HEADERS} LINKDEF LinkDef.h)
file(COPY ${HEADERS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_executable(rhtml ${SOURCES} rhtmlDict.cxx)
target_link_libraries(rhtml ROOT::Core ROOT::Rint ROOT::Gui ROOT::GuiHtml ROOT::Thread ROOT::RIO ROOT::Hist ROOT::Gpad)
set_target_properties(rhtml PROPERTIES ENABLE_EXPORTS 1)
if(MSVC AND NOT CMAKE_GENERATOR MATCHES Ninja)
  add_custom_command(TARGET rhtml POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/rhtml.exe
                                     ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_target_properties(rhtml PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif()

install(TARGETS rhtml RUNTIME DESTINATION ${PROJECT_NAME} COMPONENT applications)
install(FILES ${HEADERS} DESTINATION ${PROJECT_NAME} COMPONENT headers)
