# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT sql/odbc package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(RODBC
  HEADERS
    TODBCResult.h
    TODBCRow.h
    TODBCServer.h
    TODBCStatement.h
  SOURCES
    src/TODBCResult.cxx
    src/TODBCRow.cxx
    src/TODBCServer.cxx
    src/TODBCStatement.cxx
  DEPENDENCIES
    Net
    RIO
)

if(NOT MSVC)
  target_compile_options(RODBC PUBLIC -Wno-deprecated-declarations)
endif()
target_include_directories(RODBC PUBLIC ${ODBC_INCLUDE_DIR})
target_link_libraries(RODBC PUBLIC ${ODBC_LIBRARIES})
