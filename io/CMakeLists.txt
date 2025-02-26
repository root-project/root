# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

add_subdirectory(io)  # special CMakeLists.txt
add_subdirectory(rootpcm)
if(ROOT_ENABLE_PROJECTS AND ROOT_ENABLE_PROJECTS STREQUAL "All")
  add_subdirectory(sql)
endif()
add_subdirectory(xml)
if(xml)
  add_subdirectory(xmlparser)
endif()
if(dcache)
  add_subdirectory(dcache)
endif()
if(mpi)
  add_subdirectory(mpi)
endif()
