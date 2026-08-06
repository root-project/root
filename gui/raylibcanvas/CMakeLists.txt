# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT gui/raylibcanvas package
############################################################################

# Find raylib - expects raylib.h in include path and libraylib.a/so on library path
# On Debian/Ubuntu: sudo apt-get install libraylib-dev
# On Arch:          sudo pacman -S raylib
# Or build from source: https://github.com/raysan5/raylib

find_path(RAYLIB_INCLUDE_DIR raylib.h
  PATHS /usr/include /usr/local/include /opt/raylib/include
  DOC "raylib include directory")

find_library(RAYLIB_LIBRARY NAMES raylib raylib.lib libraylib.a libraylib.so
  PATHS /usr/lib /usr/local/lib /opt/raylib/lib
  DOC "raylib library path")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(raylib
  REQUIRED_VARS RAYLIB_LIBRARY RAYLIB_INCLUDE_DIR)

if(NOT RAYLIB_FOUND)
  if(fail-on-missing)
    message(FATAL_ERROR "Could NOT find raylib")
  else()
    message(WARNING "raylib not found, disabling option 'raylibcanvas'")
    set(raylibcanv OFF CACHE BOOL "Disabled because raylib not found" FORCE)
    return()
  endif()
endif()

# Raylib links against GLFW, OpenGL, and platform libs
# On Linux: dl, rt, m, pthread, X11, GL
set(RAYLIB_LINK_LIBRARIES ${RAYLIB_LIBRARY})
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  find_library(GLFW_LIBRARY glfw HINTS /usr/lib /usr/local/lib)
  find_library(X11_LIBRARY X11 HINTS /usr/lib /usr/local/lib)
  find_library(GL_LIBRARY GL HINTS /usr/lib /usr/local/lib)
  if(GLFW_LIBRARY)
    list(APPEND RAYLIB_LINK_LIBRARIES ${GLFW_LIBRARY})
  endif()
  if(X11_LIBRARY)
    list(APPEND RAYLIB_LINK_LIBRARIES ${X11_LIBRARY})
  endif()
  if(GL_LIBRARY)
    list(APPEND RAYLIB_LINK_LIBRARIES ${GL_LIBRARY})
  endif()
  list(APPEND RAYLIB_LINK_LIBRARIES dl rt m pthread)
endif()

set(RAYLIB_IMPORTED_TARGET raylib::raylib)
if(NOT TARGET ${RAYLIB_IMPORTED_TARGET})
  add_library(${RAYLIB_IMPORTED_TARGET} INTERFACE IMPORTED)
  set_target_properties(${RAYLIB_IMPORTED_TARGET} PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${RAYLIB_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${RAYLIB_LINK_LIBRARIES}")
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTRaylibCanvas
  HEADERS
    TRaylibCanvas.h
    TRaylibPadPainter.h
    TRaylibGuiFactory.h
  SOURCES
    src/TRaylibCanvas.cxx
    src/TRaylibPadPainter.cxx
    src/TRaylibGuiFactory.cxx
  LIBRARIES
    ${RAYLIB_IMPORTED_TARGET}
  DEPENDENCIES
    Core
    Gpad
)

target_include_directories(ROOTRaylibCanvas PRIVATE ${RAYLIB_INCLUDE_DIR})

# Platform-specific linker flags for raylib
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(ROOTRaylibCanvas PRIVATE -pthread)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  target_link_libraries(ROOTRaylibCanvas PRIVATE gdi32 winmm opengl32)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  target_link_libraries(ROOTRaylibCanvas PRIVATE "-framework Cocoa" "-framework IOKit" "-framework OpenGL")
endif()