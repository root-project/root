# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT graf3d/glew package
############################################################################

ROOT_LINKER_LIBRARY(GLEW
  src/glew.c
  LIBRARIES
    OpenGL::GL
    OpenGL::GLU
)

target_include_directories(GLEW PRIVATE ${GLEW_INCLUDE_DIR})

# Do not install headers for all platforms and configurations.
if(UNIX)
  set(installoptions FILTER "wglew")
endif()
if(NOT x11)
  set(installoptions ${installoptions} FILTER "glxew")
endif()

ROOT_INSTALL_HEADERS(${installoptions})
