############################################################################
# CMakeLists.txt file for building ROOT sql/mysql package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(RMySQL
  HEADERS
    TMySQLResult.h
    TMySQLRow.h
    TMySQLServer.h
    TMySQLStatement.h
  SOURCES
    src/TMySQLResult.cxx
    src/TMySQLRow.cxx
    src/TMySQLServer.cxx
    src/TMySQLStatement.cxx
  DICTIONARY_OPTIONS
    -I${MYSQL_INCLUDE_DIR}
  DEPENDENCIES
    Core
    Net
    RIO
)

target_include_directories(RMySQL PUBLIC ${MYSQL_INCLUDE_DIR})
target_link_libraries(RMySQL PUBLIC ${MYSQL_LIBRARIES})
