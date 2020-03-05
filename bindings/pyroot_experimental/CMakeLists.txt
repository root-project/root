# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

if(IS_ABSOLUTE ${runtimedir})
  set(d $ENV{DESTDIR}/${runtimedir})
else()
  set(d $ENV{DESTDIR}/${CMAKE_INSTALL_PREFIX}/${runtimedir})
endif()

add_subdirectory(cppyy)
add_subdirectory(pythonizations)
