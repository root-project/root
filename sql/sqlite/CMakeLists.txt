############################################################################
# CMakeLists.txt file for building ROOT sql/pgsql package
############################################################################

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
  DICTIONARY_OPTIONS
    -I${SQLITE_INCLUDE_DIR}
  DEPENDENCIES
    Core
    Net
    RIO
)

target_include_directories(RSQLite PUBLIC ${SQLITE_INCLUDE_DIR})
target_link_libraries(RSQLite PUBLIC ${SQLITE_LIBRARIES})
