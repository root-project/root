# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

add_subdirectory (geom)        # special CMakeLists.txt
add_subdirectory (geombuilder) # special CMakeLists.txt
add_subdirectory (geompainter) # special CMakeLists.txt

if(gdml)
  add_subdirectory(gdml)
  if(NOT gnuinstall)
    install(DIRECTORY gdml/ DESTINATION geom/gdml
            FILES_MATCHING PATTERN "*.py"
            PATTERN "inc" EXCLUDE
            PATTERN "src" EXCLUDE)
  endif()
endif()

if(vecgeom)
  add_subdirectory(vecgeom)
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
