include(ExternalProject)

ExternalProject_Add(BUILTIN_cppzmq
    URL https://github.com/zeromq/cppzmq/archive/refs/tags/v4.8.1.tar.gz
    URL_HASH SHA256=7a23639a45f3a0049e11a188e29aaedd10b2f4845f0000cf3e22d6774ebde0af
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    DEPENDS libzmq
)

# manually copy header (inspired by nlohmann builtin)
set(cppzmq_HEADER_PATH ${CMAKE_CURRENT_BINARY_DIR}/BUILTIN_cppzmq-prefix/src/BUILTIN_cppzmq)
add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/include/zmq.hpp
        COMMAND ${CMAKE_COMMAND} -E copy ${cppzmq_HEADER_PATH}/zmq.hpp ${CMAKE_BINARY_DIR}/include/zmq.hpp
        COMMENT "Copying zmq.hpp header to ${CMAKE_BINARY_DIR}/include"
        DEPENDS BUILTIN_cppzmq)

add_custom_target(builtin_cppzmq_incl
        DEPENDS ${CMAKE_BINARY_DIR}/include/zmq.hpp)

set_property(GLOBAL APPEND PROPERTY ROOT_HEADER_TARGETS builtin_cppzmq_incl)

install(FILES ${cppzmq_HEADER_PATH}/zmq.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/)

# find_package emulation
set(cppzmq_FOUND TRUE CACHE BOOL "" FORCE)
set(cppzmq_INCLUDE_DIR ${CMAKE_BINARY_DIR}/include CACHE INTERNAL "" FORCE)
set(cppzmq_INCLUDE_DIRS ${cppzmq_INCLUDE_DIR} CACHE INTERNAL "" FORCE)

add_library(cppzmq INTERFACE IMPORTED GLOBAL)
target_include_directories(cppzmq INTERFACE $<BUILD_INTERFACE:${cppzmq_INCLUDE_DIR}>)
add_dependencies(cppzmq BUILTIN_cppzmq builtin_cppzmq_incl)
