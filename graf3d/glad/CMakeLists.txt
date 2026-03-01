# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT graph3d/rglad package
############################################################################

# This package is only needed for user backward compatibility! (TGLIncludes.h)

set(rglad_srcs src/gl.c)

find_library(DL_LIBRARY_PATH dl)
if (DL_LIBRARY_PATH)
  list(APPEND rglad_libs dl)
endif()

if(x11)
  list(APPEND rglad_srcs src/glx.c)
  list(APPEND rglad_libs ${X11_LIBRARIES})
endif()

if(MSVC)
  list(APPEND rglad_srcs src/wgl.c)
  list(APPEND rglad_libs OpenGL32.lib)
endif()

ROOT_LINKER_LIBRARY(ROOTGlad ${rglad_srcs} LIBRARIES ${rglad_libs})
target_compile_definitions(ROOTGlad
  PRIVATE
    GLAD_API_CALL_EXPORT_BUILD
  PUBLIC
    GLAD_API_CALL_EXPORT
)
