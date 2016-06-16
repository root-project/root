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

#---For the windows platform ODBC is located automatically
if(WIN32)
  set(ODBC_INCLUDE_DIR "")
  set(ODBC_LIBRARY odbc32.lib)
else()
  find_path(ODBC_INCLUDE_DIR sqlext.h
    PATH_SUFFIXES odbc iodbc 
    HINTS $ENV{ODBC_DIR}/include ${ODBC_DIR}/include 
    DOC "Specify the directory containing sql.h."
  )

  find_library( ODBC_LIBRARY NAMES iodbc odbc odbc32
    PATHS_SUFFIXES odbc
    HINTS  $ENV{ODBC_DIR}/lib ${ODBC_DIR}/lib 
    DOC "Specify the ODBC driver manager library here."
  )
endif()

set(ODBC_LIBRARIES ${ODBC_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ODBC DEFAULT_MSG ODBC_INCLUDE_DIR ODBC_LIBRARY)
mark_as_advanced(ODBC_INCLUDE_DIR OBC_LIBRARY)

