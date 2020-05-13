# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT sql/pgsql package
############################################################################

if(runtime_cxxmodules AND APPLE)
  list(APPEND extra_dictionary_opts "-mByproduct" "SQLite3")
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(RSQLite
  HEADERS
    TSQLiteResult.h
    TSQLiteRow.h
    TSQLiteServer.h
    TSQLiteStatement.h
  SOURCES
    src/TSQLiteResult.cxx
    src/TSQLiteRow.cxx
    src/TSQLiteServer.cxx
    src/TSQLiteStatement.cxx
  DEPENDENCIES
    Core
    Net
    RIO
  DICTIONARY_OPTIONS
    ${extra_dictionary_opts}
)

target_include_directories(RSQLite PUBLIC ${SQLITE_INCLUDE_DIR})
target_link_libraries(RSQLite PUBLIC ${SQLITE_LIBRARIES})
