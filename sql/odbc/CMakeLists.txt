############################################################################
# CMakeLists.txt file for building ROOT sql/odbc package
############################################################################

include_directories(${ODBC_INCLUDE_DIR})
if(MACOSX_ODBC_DEPRECATED)
  add_definitions(-Wno-deprecated-declarations)
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(RODBC
                              LIBRARIES ${ODBC_LIBRARIES}
                              DEPENDENCIES Net RIO)
