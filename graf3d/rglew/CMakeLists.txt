# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT graph3d/rglew package
############################################################################

# This package is only needed for user backward compatibility! (TGLIncludes.h)

# Figure out how not to install wgl / glx headers and to not try to
# compile the corresponding source files.
# For now hardcode for linux.

if(x11)
  ROOT_LINKER_LIBRARY(RGlew
    src/gl.c
    src/glx.c
  LIBRARIES
    OpenGL::GL
    ${X11_LIBRARIES}
  )
else()
  ROOT_LINKER_LIBRARY(RGlew
    src/gl.c
  #  src/wgl.c
  LIBRARIES
    OpenGL::GL
  )
endif()

# Is this needed?
target_include_directories(RGlew PRIVATE   ${CMAKE_SOURCE_DIR}/graf3d/rglew/inc/)

# We still need to install TGLIncludes.h into include/
ROOT_INSTALL_HEADERS()
