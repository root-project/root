# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# Find the FastCGI includes and library
#
#  FASTCGI_INCLUDE_DIR - where to find fcgiapp.h
#  FASTCGI_LIBRARY     - library when using FastCGI.
#  FASTCGI_FOUND       - true if FASTCGI found.

if(NOT FASTCGI_INCLUDE_DIR)
   find_path(FASTCGI_INCLUDE_DIR NAME fcgiapp.h PATH_SUFFIXES include)
endif()

if(NOT FASTCGI_LIBRARY)
   find_library(FASTCGI_LIBRARY NAMES fcgi PATHS PATH_SUFFIXES lib)
endif()

mark_as_advanced(FASTCGI_INCLUDE_DIR)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(FASTCGI
  REQUIRED_VARS FASTCGI_LIBRARY FASTCGI_INCLUDE_DIR)
