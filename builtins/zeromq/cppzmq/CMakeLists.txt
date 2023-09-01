include(ExternalProject)

set(cppzmq_HEADER_PATH ${CMAKE_CURRENT_BINARY_DIR}/BUILTIN_cppzmq-prefix/src/BUILTIN_cppzmq)

set(lcgpackages http://lcgpackages.web.cern.ch/lcgpackages/tarFiles/sources)
ExternalProject_Add(BUILTIN_cppzmq
    URL ${lcgpackages}/cppzmq-4.8.1.tar.gz
    URL_HASH SHA256=7a23639a45f3a0049e11a188e29aaedd10b2f4845f0000cf3e22d6774ebde0af
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
