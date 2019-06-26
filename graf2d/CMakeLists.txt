# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

if(root7)
  add_subdirectory(primitives)
  add_subdirectory(gpadv7)
endif()
add_subdirectory(gpad)
add_subdirectory(graf)
add_subdirectory(postscript)
add_subdirectory(mathtext)
if(WIN32)
  add_subdirectory(win32gdk)
elseif(x11)
  add_subdirectory(x11)         # special CMakeLists.txt
  add_subdirectory(x11ttf)      # special CMakeLists.txt
endif()
if(asimage)
  add_subdirectory(asimage)     # special CMakeLists.txt
endif()
if(gviz)
  add_subdirectory(gviz)
endif()
if(fitsio)
  add_subdirectory(fitsio)
endif()
if(cocoa)
  add_subdirectory(quartz)
  add_subdirectory(cocoa)
endif()
