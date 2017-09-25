############################################################################
# CMakeLists.txt file for building ROOT sql/pgsql package
############################################################################

include_directories(${SQLITE_INCLUDE_DIR})

ROOT_STANDARD_LIBRARY_PACKAGE(RSQLite
                              LIBRARIES Core ${SQLITE_LIBRARIES}
                              DEPENDENCIES Core Net RIO)
