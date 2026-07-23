# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

add_subdirectory(g3d)

if(NOT WIN32 AND x11)
  add_subdirectory(x3d)
endif()

add_subdirectory(csg)

if (opengl)
  add_subdirectory(gl)
endif()

if(geom)
  add_subdirectory(TGeometry)

  if(opengl)
    add_subdirectory(eve)
    add_subdirectory(gviz3d)
  endif()

  if(webgui AND root7)
    add_subdirectory(eve7)
  endif()
endif()
