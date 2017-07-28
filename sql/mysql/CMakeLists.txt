############################################################################
# CMakeLists.txt file for building ROOT sql/mysql package
############################################################################

include_directories(${MYSQL_INCLUDE_DIR})

ROOT_STANDARD_LIBRARY_PACKAGE(RMySQL
                              LIBRARIES ${MYSQL_LIBRARIES}
                              DEPENDENCIES Core Net RIO)
