# Find the native MySQL includes and library
#
#  MYSQL_INCLUDE_DIR - where to find mysql.h, etc.
#  MYSQL_LIBRARIES   - List of libraries when using MySQL.
#  MYSQL_FOUND       - True if MySQL found.

if(MYSQL_INCLUDE_DIR OR MYSQL_)
  # Already in cache, be silent
  SET(MYSQL_FIND_QUIETLY TRUE)
endif()

if(NOT WIN32)
  find_program(MYSQL_CONFIG_EXECUTABLE mysql_config
    /usr/bin/
    /usr/local/bin
    ${MYSQL_DIR}/bin $ENV{MYSQL_DIR}/bin
  )
endif()

if(MYSQL_CONFIG_EXECUTABLE)
  execute_process(COMMAND ${MYSQL_CONFIG_EXECUTABLE} --cflags OUTPUT_VARIABLE MYSQL_CFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
  separate_arguments(MYSQL_CFLAGS)
  string( REGEX MATCHALL "-I[^;]+" MYSQL_INCLUDE_DIRS "${MYSQL_CFLAGS}" )
  string( REPLACE "-I" "" MYSQL_INCLUDE_DIRS "${MYSQL_INCLUDE_DIRS}")
  find_path(MYSQL_INCLUDE_DIR mysql.h PATHS ${MYSQL_INCLUDE_DIRS} NO_DEFAULT_PATH)
  execute_process(COMMAND ${MYSQL_CONFIG_EXECUTABLE} --libs OUTPUT_VARIABLE MYSQL_LIBRARIES OUTPUT_STRIP_TRAILING_WHITESPACE)
else()
  find_path(MYSQL_INCLUDE_DIR mysql.h
    /usr/local/mysql/include
    /usr/local/include/mysql
    /usr/local/include
    /usr/include/mysql
    /usr/include
    /usr/mysql/include
    $ENV{MYSQL_DIR}/include
  )
  set(MYSQL_NAMES mysqlclient mysqlclient_r)
  find_library(MYSQL_LIBRARY NAMES ${MYSQL_NAMES}
    PATHS /usr/local/mysql/lib /usr/local/lib /usr/lib $ENV{MYSQL_DIR}/lib $ENV{MYSQL_DIR}/lib/opt
  )
  set(MYSQL_LIBRARIES ${MYSQL_LIBRARY})
endif()

# handle the QUIETLY and REQUIRED arguments and set DCAP_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MYSQL DEFAULT_MSG MYSQL_INCLUDE_DIR MYSQL_LIBRARIES)

mark_as_advanced(
  MYSQL_CONFIG_EXECUTABLE
  MYSQL_LIBRARY
  MYSQL_INCLUDE_DIR
)

