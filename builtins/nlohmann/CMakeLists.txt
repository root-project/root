# Install nlohmann/json.hpp and json_fwd.hpp
# This file will define the target
#   nlohmann_json
# and the alias
#   nlohmann_json::nlohmann_json
# with proper #defines and includes. Use the alias target with the full prefix for access to JSON.

# The installed files are only used when ACLiC or ROOT macros will include REve headers,
# they are not used for ROOT compilation, as this happens directly from the source directory.

# extract version from existing header file
file(STRINGS "json.hpp" JSON_H REGEX "^#define NLOHMANN_JSON_VERSION_[A-Z]+[ ]+[0-9]+.*$")
string(REGEX REPLACE ".+NLOHMANN_JSON_VERSION_MAJOR[ ]+([0-9]+).*$"   "\\1" JSON_VERSION_MAJOR "${JSON_H}")
string(REGEX REPLACE ".+NLOHMANN_JSON_VERSION_MINOR[ ]+([0-9]+).*$"   "\\1" JSON_VERSION_MINOR "${JSON_H}")
string(REGEX REPLACE ".+NLOHMANN_JSON_VERSION_PATCH[ ]+([0-9]+).*$" "\\1" JSON_VERSION_PATCH "${JSON_H}")
set(nlohmann_json_VERSION "${JSON_VERSION_MAJOR}.${JSON_VERSION_MINOR}.${JSON_VERSION_PATCH}" PARENT_SCOPE)
unset(JSON_H)

add_custom_command(
     OUTPUT ${CMAKE_BINARY_DIR}/include/nlohmann/json.hpp
     COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/builtins/nlohmann/json.hpp ${CMAKE_BINARY_DIR}/include/nlohmann/json.hpp
     DEPENDS ${CMAKE_SOURCE_DIR}/builtins/nlohmann/json.hpp)
add_custom_target(builtin_nlohmann_json_incl DEPENDS ${CMAKE_BINARY_DIR}/include/nlohmann/json.hpp)
set_property(GLOBAL APPEND PROPERTY ROOT_BUILTIN_TARGETS builtin_nlohmann_json_incl)

install(FILES ${CMAKE_SOURCE_DIR}/builtins/nlohmann/json.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nlohmann/)

# Create a target and all its configs:
add_library(nlohmann_json INTERFACE)

target_include_directories(nlohmann_json INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/builtins> $<INSTALL_INTERFACE:include>)
target_compile_definitions(nlohmann_json INTERFACE NLOHMANN_JSON_PROVIDES_FWD_HPP)

install(TARGETS nlohmann_json
  EXPORT ROOTExports)
set_property(GLOBAL APPEND PROPERTY ROOT_EXPORTED_TARGETS nlohmann_json)

# Alias, so can use it as drop-in replacement for system nlohmann_json.
add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)
