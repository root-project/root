include(ExternalProject)

set(cppzmq_HEADER_PATH ${CMAKE_CURRENT_BINARY_DIR}/BUILTIN_cppzmq-prefix/src/BUILTIN_cppzmq)

set(lcgpackages http://lcgpackages.web.cern.ch/lcgpackages/tarFiles/sources)
set(cppzmq_version 4.10.0)
ExternalProject_Add(BUILTIN_cppzmq
    URL ${lcgpackages}/cppzmq-${cppzmq_version}.tar.gz
    URL_HASH SHA256=c81c81bba8a7644c84932225f018b5088743a22999c6d82a2b5f5cd1e6942b74
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    DEPENDS libzmq
    BUILD_BYPRODUCTS ${cppzmq_HEADER_PATH}/zmq.hpp
)

# find_package emulation
set(cppzmq_FOUND TRUE CACHE BOOL "" FORCE)
set(cppzmq_INCLUDE_DIR ${cppzmq_HEADER_PATH} CACHE INTERNAL "" FORCE)
set(cppzmq_INCLUDE_DIRS ${cppzmq_INCLUDE_DIR} CACHE INTERNAL "" FORCE)

add_library(cppzmq INTERFACE IMPORTED GLOBAL)
target_include_directories(cppzmq INTERFACE $<BUILD_INTERFACE:${cppzmq_INCLUDE_DIR}>)
add_dependencies(cppzmq BUILTIN_cppzmq)
