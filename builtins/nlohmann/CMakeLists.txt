# Install nlohmann/json.hpp include to have it

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


