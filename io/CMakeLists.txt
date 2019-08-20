# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

add_subdirectory(io)  # special CMakeLists.txt
add_subdirectory(rootpcm)
add_subdirectory(sql)
add_subdirectory(xml)
if(xml)
  add_subdirectory(xmlparser)
endif()
if(gfal)
  add_subdirectory(gfal)
endif()
if(dcache)
  add_subdirectory(dcache)
endif()
if(mpi)
  add_subdirectory(mpi)
endif()
