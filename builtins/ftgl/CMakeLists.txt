# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# **PLEASE UPDATE ALSO THE FOLLOWING LINE WHEN UPDATING THE VERSION**
# 7 Feb 2019, https://github.com/frankheckenbach/ftgl/releases/tag/v2.4.0
set(ROOT_FTGL_VERSION 2.4.0)
set(ROOT_FTGL_HASH "aa97da1c3442a8fd3941037655df18016d70b5266381c81d81e8b5335f196ea8")
if (CMAKE_BUILD_TYPE MATCHES Debug)
  set(ROOT_FTGL_POSTFIX d)
endif()
# Cherry-pick https://github.com/frankheckenbach/ftgl/commit/835f2ba7911a6c15a1a314d5e3267fa089b5a319 :
set(ROOT_FTGL_PATCH_FILE_1 "${CMAKE_CURRENT_SOURCE_DIR}/0001-fix-CMakeLists.patch")
# From https://github.com/frankheckenbach/ftgl/issues/22, see also https://github.com/xmake-io/xmake-repo/pull/9040/changes and https://github.com/root-project/root/commit/a028d13255f0c2d0c84f2e1b99ccb6a112598e0c
if(Freetype_VERSION VERSION_GREATER_EQUAL 2.13.3) # https://github.com/root-project/root/commit/a028d13255f0c2d0c84f2e1b99ccb6a112598e0c
  set(ROOT_FTGL_PATCH_FILE_2 "${CMAKE_CURRENT_SOURCE_DIR}/0002-fix-type-error.patch")
else()
  set(ROOT_FTGL_PATCH_FILE_2 "")
endif()
set(ROOT_FTGL_PATCH_FILE_3 "${CMAKE_CURRENT_SOURCE_DIR}/0003-fix-nullptr.patch") # https://github.com/root-project/root/issues/22076#issuecomment-4335481094
set(ROOT_FTGL_PREFIX ${CMAKE_BINARY_DIR}/builtins/FTGL-prefix)
set(ROOT_FTGL_LIBRARY ${ROOT_FTGL_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}ftgl${ROOT_FTGL_POSTFIX}${CMAKE_STATIC_LIBRARY_SUFFIX})

if (NOT DEFINED GIT_EXECUTABLE)
  set(GIT_EXECUTABLE "git")
endif()

if(NOT EXISTS "${ROOT_FTGL_PREFIX}/src/BUILTIN_FTGL/._patched")
  set(ROOT_FTGL_PATCH_COMMAND PATCH_COMMAND ${GIT_EXECUTABLE} apply --ignore-space-change --ignore-whitespace
                ${ROOT_FTGL_PATCH_FILE_1}
                ${ROOT_FTGL_PATCH_FILE_2}
                ${ROOT_FTGL_PATCH_FILE_3}
                ${CMAKE_CURRENT_SOURCE_DIR}/ftgl-marker.diff)
endif()

include(ExternalProject)

# Clear cache variables set by find_package(FTGL)
# to ensure that we use the builtin version
foreach(var FTGL_LIBRARIES FTGL_LIBRARY FTGL_LIBRARY_DEBUG FTGL_LIBRARY_RELEASE FTGL_FOUND FTGL_VERSION FTGL_INCLUDE_DIR FTGL_LIBRARY FTGL_LIBRARIES)
  unset(${var})
  unset(${var} CACHE)
endforeach()

if(WIN32 AND NOT CMAKE_GENERATOR MATCHES Ninja)
  if(winrtdebug)
    set(ROOT_FTGL_BUILD_COMMAND_FLAGS "--config Debug")
  else()
    set(ROOT_FTGL_BUILD_COMMAND_FLAGS "--config $<IF:$<CONFIG:Debug,RelWithDebInfo>,RelWithDebInfo,Release>")
  endif()
endif()

ExternalProject_Add(BUILTIN_FTGL
  PREFIX ${ROOT_FTGL_PREFIX}
  URL ${lcgpackages}/ftgl-${ROOT_FTGL_VERSION}.tar.gz
  URL_HASH SHA256=${ROOT_FTGL_HASH}
  ${ROOT_FTGL_PATCH_COMMAND}

  LOG_DOWNLOAD  TRUE
  LOG_CONFIGURE TRUE
  LOG_BUILD     TRUE
  LOG_PATCH     TRUE
  LOG_INSTALL   TRUE
  LOG_OUTPUT_ON_FAILURE TRUE

  CMAKE_ARGS -G ${CMAKE_GENERATOR}
    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
    -DBUILD_SHARED_LIBS:BOOL=FALSE
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=TRUE
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    -DFREETYPE_LIBRARY=${FREETYPE_LIBRARIES} # Seems to be needed for Win32 (CMake 3.30 Freetype.cmake)
    -DFREETYPE_INCLUDE_DIRS=${FREETYPE_INCLUDE_DIRS} # Seems to be needed for Win32 (CMake 3.30 Freetype.cmake)
    -DFREETYPE_DIR=${FREETYPE_INCLUDE_DIRS}/../ # Seems to be needed for Win32 (CMake 3.30 Freetype.cmake)
  BUILD_COMMAND ${CMAKE_COMMAND} --build . ${ROOT_FTGL_BUILD_COMMAND_FLAGS}
  INSTALL_COMMAND ${CMAKE_COMMAND} --build . ${ROOT_FTGL_BUILD_COMMAND_FLAGS} --target install
  BUILD_BYPRODUCTS
    ${ROOT_FTGL_LIBRARY}
  TIMEOUT 600
)

file(MAKE_DIRECTORY ${ROOT_FTGL_PREFIX}/include)
add_library(FTGL::FTGL IMPORTED STATIC GLOBAL)
add_dependencies(FTGL::FTGL BUILTIN_FTGL)
set_target_properties(FTGL::FTGL PROPERTIES
    IMPORTED_LOCATION ${ROOT_FTGL_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${ROOT_FTGL_PREFIX}/include)
target_compile_definitions(FTGL::FTGL INTERFACE FTGL_LIBRARY_STATIC) # needed for Win32 since public flag is not correctly propagated to parent scope (BUILD_SHARED_LIBS works fine for building but when installing, flag info is lost)
if(builtin_freetype)
  add_dependencies(BUILTIN_FTGL BUILTIN_FREETYPE)
endif()
if(builin_zlib)
  add_dependencies(BUILTIN_FTGL BUILTIN_ZLIB)
endif()
target_link_libraries(FTGL::FTGL INTERFACE Freetype::Freetype) # private: OpenGL::GL OpenGL::GLU ZLIB::ZLIB

# Set the canonical output of find_package according to
# https://cmake.org/cmake/help/latest/manual/cmake-developer.7.html#standard-variable-names
set(FTGL_INCLUDE_DIRS ${ROOT_FTGL_PREFIX}/include PARENT_SCOPE)
set(FTGL_LIBRARIES ${ROOT_FTGL_LIBRARY} PARENT_SCOPE)
set(FTGL_FOUND TRUE PARENT_SCOPE)
set(FTGL_VERSION ${ROOT_FTGL_VERSION} PARENT_SCOPE)
