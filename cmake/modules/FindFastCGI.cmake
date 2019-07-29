# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# Find the FastCGI includes and library
#
#  FASTCGI_INCLUDE_DIR - where to find fcgiapp.h
#  FASTCGI_LIBRARY     - librart when using MySQL.
#  FASTCGI_FOUND       - True if FASTCGI found.

if(FASTCGI_INCLUDE_DIR OR FASTCGI_)
  # Already in cache, be silent
  SET(FASTCGI_FIND_QUIETLY TRUE)
endif()

find_path(FASTCGI_INCLUDE_DIR fcgiapp.h
  $ENV{FASTCGI_DIR}/include
  /usr/local/include
  /usr/include/fastcgi
  /usr/local/include/fastcgi
  /opt/fastcgi/include
  DOC "Specify the directory containing fcgiapp.h"
)

find_library(FASTCGI_LIBRARY NAMES fcgi PATHS
  $ENV{FASTCGI_DIR}/lib
  /usr/local/fastcgi/lib
  /usr/local/lib
  /usr/lib/fastcgi
  /usr/local/lib/fastcgi
  /usr/fastcgi/lib /usr/lib
  /usr/fastcgi /usr/local/fastcgi
  /opt/fastcgi /opt/fastcgi/lib
  DOC "Specify the FastCGI library here."
)

# handle the QUIETLY and REQUIRED arguments and set DCAP_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FASTCGI DEFAULT_MSG FASTCGI_INCLUDE_DIR FASTCGI_LIBRARY)

mark_as_advanced(
  FASTCGI_LIBRARY
  FASTCGI_INCLUDE_DIR
)
