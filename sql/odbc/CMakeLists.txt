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
  DICTIONARY_OPTIONS
    -I${ODBC_INCLUDE_DIR}
  DEPENDENCIES
    Net
    RIO
)

target_compile_options(RODBC PUBLIC -Wno-deprecated-declarations)
target_include_directories(RODBC PUBLIC ${ODBC_INCLUDE_DIR})
target_link_libraries(RODBC PUBLIC ${ODBC_LIBRARIES})
