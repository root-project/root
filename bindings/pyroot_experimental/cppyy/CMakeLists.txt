# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

if(MSVC)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".dll")
else()
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
endif()

add_subdirectory(cppyy-backend)
add_subdirectory(CPyCppyy)
add_subdirectory(cppyy)
