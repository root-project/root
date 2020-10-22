# Fictional nlohman_json library target to get proper include path

add_library(nlohmann_json INTERFACE)
add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)

target_include_directories(nlohmann_json INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/builtins>)

# file only used when ACLiC or ROOT macros will include REve headers,
# it is not used for ROOT compilation

add_custom_command(
     OUTPUT ${CMAKE_BINARY_DIR}/include/nlohmann/json.hpp
     COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/builtins/nlohmann/json.hpp ${CMAKE_BINARY_DIR}/include/nlohmann/json.hpp
     COMMENT "Copying nlohmann/json.hpp header to ${CMAKE_BINARY_DIR}/include"
     DEPENDS ${CMAKE_SOURCE_DIR}/builtins/nlohmann/json.hpp)

add_custom_target(builtin_nlohmann_json_incl DEPENDS ${CMAKE_BINARY_DIR}/include/nlohmann/json.hpp)

set_property(GLOBAL APPEND PROPERTY ROOT_HEADER_TARGETS builtin_nlohmann_json_incl)

install(FILES ${CMAKE_SOURCE_DIR}/builtins/nlohmann/json.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nlohmann/)

export(TARGETS nlohmann_json
        NAMESPACE nlohmann_json::
        FILE ${CMAKE_CURRENT_BINARY_DIR}/nlohmann_jsonTargets.cmake)

install(TARGETS nlohmann_json
        EXPORT nlohmann_jsonExports
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT nlohmann_jsonExports
        NAMESPACE nlohmann_json::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nlohmann_json)