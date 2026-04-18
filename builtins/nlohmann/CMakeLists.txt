# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# **PLEASE UPDATE ALSO THE FOLLOWING LINE WHEN UPDATING THE VERSION**
# 11 Nov 2025, https://github.com/nlohmann/json/releases/tag/v3.12.0
set(ROOT_NLOHMANN_VERSION 3.12.0)
set(ROOT_NLOHMANN_HASH "4b92eb0c06d10683f7447ce9406cb97cd4b453be18d7279320f7b2f025c10187")
set(ROOT_NLOHMANN_PREFIX ${CMAKE_BINARY_DIR}/builtins/NLOHMANN-prefix)

unset(nlohmann_json_DIR)
unset(nlohmann_json_DIR CACHE)

if(WIN32 AND NOT CMAKE_GENERATOR MATCHES Ninja)
  if(winrtdebug)
    set(ROOT_NLOHMANN_BUILD_COMMAND_FLAGS "--config Debug")
  else()
    set(ROOT_NLOHMANN_BUILD_COMMAND_FLAGS "--config $<IF:$<CONFIG:Debug,RelWithDebInfo>,RelWithDebInfo,Release>")
  endif()
endif()

ExternalProject_Add(
  BUILTIN_NLOHMANN
  PREFIX ${ROOT_NLOHMANN_PREFIX}
  URL ${lcgpackages}/json-${ROOT_NLOHMANN_VERSION}.tar.gz
  URL_HASH SHA256=${ROOT_NLOHMANN_HASH}
  CMAKE_ARGS -G ${CMAKE_GENERATOR}
             -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
             -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
             -DBUILD_TESTING=OFF
             -DJSON_32bitTest=OFF
             -DJSON_BuildTests=OFF
             -DJSON_MultipleHeaders=OFF
  BUILD_COMMAND ${CMAKE_COMMAND} --build . ${ROOT_NLOHMANN_BUILD_COMMAND_FLAGS}
  INSTALL_COMMAND ${CMAKE_COMMAND} --build . ${ROOT_NLOHMANN_BUILD_COMMAND_FLAGS} --target install
  LOG_DOWNLOAD 1
  LOG_CONFIGURE 1
  LOG_BUILD 1
  LOG_INSTALL 1
  LOG_OUTPUT_ON_FAILURE 1
  TIMEOUT 600)

file(MAKE_DIRECTORY ${ROOT_NLOHMANN_PREFIX}/include)
add_library(nlohmann_json::nlohmann_json IMPORTED INTERFACE GLOBAL)
add_dependencies(nlohmann_json::nlohmann_json BUILTIN_NLOHMANN)
set_target_properties(nlohmann_json::nlohmann_json PROPERTIES
  INTERFACE_COMPILE_FEATURES "cxx_std_11"
  INTERFACE_INCLUDE_DIRECTORIES ${ROOT_NLOHMANN_PREFIX}/include)
target_compile_definitions(nlohmann_json::nlohmann_json INTERFACE NLOHMANN_JSON_PROVIDES_FWD_HPP)

# Set the canonical output of find_package according to
# https://cmake.org/cmake/help/latest/manual/cmake-developer.7.html#standard-variable-names
set(nlohmann_json_INCLUDE_DIRS ${ROOT_NLOHMANN_PREFIX}/include PARENT_SCOPE)
set(nlohmann_json_FOUND TRUE PARENT_SCOPE)
set(nlohmann_json_VERSION ${ROOT_NLOHMANN_VERSION} PARENT_SCOPE)
