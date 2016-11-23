# - Find PostgreSQL
# Find the PostgreSQL includes and client library
# This module defines
#  POSTGRESQL_INCLUDE_DIR, where to find POSTGRESQL.h
#  POSTGRESQL_LIBRARIES, the libraries needed to use POSTGRESQL.
#  POSTGRESQL_FOUND, If false, do not try to use PostgreSQL.

# Copyright (c) 2006, Jaroslaw Staniek, <js@iidea.pl>
#
# Redistribution and use is allowed according to the terms of the BSD license.
# For details see the accompanying COPYING-CMAKE-SCRIPTS file.


find_path(POSTGRESQL_INCLUDE_DIR libpq-fe.h
   HINTS ${POSTGRESQL_DIR}/include $ENV{POSTGRESQL_DIR}/include
   PATH_SUFFIXES pgsql pgsql/server postgresql postgresql/server
)

find_library(POSTGRESQL_LIBRARY NAMES pq
    HINTS ${POSTGRESQL_DIR}/lib $ENV{POSTGRESQL_DIR}/lib
    PATH_SUFFIXES pgsql posgresql
)

if(POSTGRESQL_LIBRARY)
  set(POSTGRESQL_LIBRARIES ${POSTGRESQL_LIBRARY})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PostgreSQL DEFAULT_MSG
                                  POSTGRESQL_INCLUDE_DIR POSTGRESQL_LIBRARY)

mark_as_advanced(POSTGRESQL_INCLUDE_DIR POSTGRESQL_LIBRARY)
