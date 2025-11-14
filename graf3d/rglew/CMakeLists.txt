# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT graph3d/rglew package
############################################################################

# This package is only needed for user backward compatibility! (TGLIncludes.h)

set(rglew_srcs src/gl.c)

find_library(DL_LIBRARY_PATH dl)
if (DL_LIBRARY_PATH)
  list(APPEND rglew_libs dl)
endif()

if(x11)
  list(APPEND rglew_srcs src/glx.c)
  list(APPEND rglew_libs ${X11_LIBRARIES})
endif()

if(MSVC)
  list(APPEND rglew_srcs src/wgl.c)
  list(APPEND rglew_libs OpenGL32.lib)
endif()

ROOT_LINKER_LIBRARY(RGlew ${rglew_srcs} LIBRARIES ${rglew_libs})
if(MSVC)
  target_compile_definitions(RGlew PRIVATE GLAD_API_CALL_EXPORT GLAD_API_CALL_EXPORT_BUILD)
endif()

# We still need to install TGLIncludes.h into include/
ROOT_INSTALL_HEADERS()
