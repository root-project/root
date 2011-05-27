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


if (POSTGRESQL_INCLUDE_DIR AND POSTGRESQL_LIBRARIES)
  # Already in cache, be silent
  set(PostgreSQL_FIND_QUIETLY TRUE)
endif (POSTGRESQL_INCLUDE_DIR AND POSTGRESQL_LIBRARIES)


find_path(POSTGRESQL_INCLUDE_DIR libpq-fe.h
   /usr/include/pgsql/
   /usr/local/include/pgsql/
   /usr/include/postgresql/
)

find_library(POSTGRESQL_LIBRARIES NAMES pq)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PostgreSQL DEFAULT_MSG
                                  POSTGRESQL_INCLUDE_DIR POSTGRESQL_LIBRARIES )

mark_as_advanced(POSTGRESQL_INCLUDE_DIR POSTGRESQL_LIBRARIES)

