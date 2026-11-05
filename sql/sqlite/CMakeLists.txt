# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT sql/sqlite package
############################################################################

set(RSQLITE_HEADERS
  TSQLiteResult.h
  TSQLiteRow.h
  TSQLiteServer.h
  TSQLiteStatement.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(RSQLite
  HEADERS
    ${RSQLITE_HEADERS}
  SOURCES
    src/TSQLiteResult.cxx
    src/TSQLiteRow.cxx
    src/TSQLiteServer.cxx
    src/TSQLiteStatement.cxx
  DEPENDENCIES
    Core
    Net
    RIO
)

set(RELATIVE_INC_HEADERS ${RSQLITE_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(RSQLite
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)

target_include_directories(RSQLite SYSTEM PUBLIC ${SQLITE_INCLUDE_DIR})
target_link_libraries(RSQLite PUBLIC ${SQLITE_LIBRARIES})
ROOT_ADD_TEST_SUBDIRECTORY(test)
