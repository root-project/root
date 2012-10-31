# Find the ODBC driver manager includes and library.
# 
# ODBC is an open standard for connecting to different databases in a
# semi-vendor-independent fashion.  First you install the ODBC driver
# manager.  Then you need a driver for each separate database you want
# to connect to (unless a generic one works).  VTK includes neither
# the driver manager nor the vendor-specific drivers: you have to find
# those yourself.
#  
# This module defines
# ODBC_INCLUDE_DIR where to find sql.h
# ODBC_LIBRARIES, the libraries to link against to use ODBC
# ODBC_FOUND.  If false, you cannot build anything that requires MySQL.
# also defined, but not for general use is
# ODBC_LIBRARY, where to find the ODBC driver manager library.

SET( ODBC_FOUND 0 )

#---For the windows platform ODBC is located automatically
if(WIN32)
  set(ODBC_INCLUDE_DIR "")
  set(ODBC_LIBRARY odbc32.lib) 
  set(ODBC_FOUND 1)
else()
  find_path(ODBC_INCLUDE_DIR sqlext.h
    /usr/include
    /usr/include/odbc
    /usr/local/include
    /usr/local/include/odbc
    /usr/local/odbc/include
	$ENV{ODBC_DIR}/include
    DOC "Specify the directory containing sql.h."
  )

  find_library( ODBC_LIBRARY NAMES iodbc odbc odbc32
    PATHS
    /usr/lib
    /usr/lib/odbc
    /usr/local/lib
    /usr/local/lib/odbc
    /usr/local/odbc/lib
	$ENV{ODBC_DIR}/lib
    DOC "Specify the ODBC driver manager library here."
  )
  if(ODBC_LIBRARY AND ODBC_INCLUDE_DIR)
    set( ODBC_FOUND 1 )
  endif()
endif()


set(ODBC_LIBRARIES ${ODBC_LIBRARY})

MARK_AS_ADVANCED( ODBC_FOUND ODBC_LIBRARY ODBC_EXTRA_LIBRARIES ODBC_INCLUDE_DIR )
