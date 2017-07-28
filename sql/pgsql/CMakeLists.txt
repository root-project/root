############################################################################
# CMakeLists.txt file for building ROOT sql/pgsql package
############################################################################

include_directories(${POSTGRESQL_INCLUDE_DIR})

ROOT_STANDARD_LIBRARY_PACKAGE(PgSQL
                              LIBRARIES ${POSTGRESQL_LIBRARIES}
                              DEPENDENCIES Core Net RIO)
