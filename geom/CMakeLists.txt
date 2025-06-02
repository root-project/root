# Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

add_subdirectory (geom)        # special CMakeLists.txt
if(geombuilder)
  add_subdirectory (geombuilder) # special CMakeLists.txt
endif()

add_subdirectory(geomchecker)

add_subdirectory(geompainter)

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

if(webgui)
  add_subdirectory(webviewer)
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
