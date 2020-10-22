# Fuctional nlohman_json library target to get proper dependency

add_library(nlohmann_json INTERFACE)
add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)

target_include_directories(nlohmann_json INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/builtins>)

# configure_file(json.hpp ${CMAKE_BINARY_DIR}/ginclude/nlohmann/json.hpp COPYONLY)

install(FILES ${CMAKE_SOURCE_DIR}/builtins/nlohmann/json.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nlohmann/)

set_property(GLOBAL APPEND PROPERTY ROOT_BUILTIN_TARGETS nlohmann_json::nlohmann_json)


export(TARGETS nlohmann_json
        NAMESPACE nlohmann_json::
        FILE ${CMAKE_CURRENT_BINARY_DIR}/nlohmann_jsonTargets.cmake)

install(TARGETS nlohmann_json
        EXPORT nlohmann_jsonExports
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT nlohmann_jsonExports
        NAMESPACE nlohmann_json::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nlohmann_json)