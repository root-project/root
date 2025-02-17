# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

add_subdirectory(g3d) # special CMakeLists.txt
if(NOT WIN32 AND x11)
  add_subdirectory(x3d) # special CMakeLists.txt
endif()

add_subdirectory(csg) # special CMakeLists.txt

if (opengl)
  if(builtin_ftgl)
    add_subdirectory(ftgl)
  endif()
  add_subdirectory(rglew)
  if(geom)
    add_subdirectory(eve)
    add_subdirectory(gviz3d)
  endif()
  add_subdirectory(gl)
endif()

if(webgui AND root7 AND geom)
  add_subdirectory(eve7) # special CMakeLists.txt
endif()
