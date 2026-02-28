include(ExternalProject)

set(ZeroMQ_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/BUILTIN_ZeroMQ-prefix)
set(ZeroMQ_LIBNAME ${CMAKE_STATIC_LIBRARY_PREFIX}zmq${CMAKE_STATIC_LIBRARY_SUFFIX})

set(ZeroMQ_LIBRARY ${ZeroMQ_PREFIX}/lib/${ZeroMQ_LIBNAME} CACHE INTERNAL "" FORCE)
set(ZeroMQ_LIBRARIES ${ZeroMQ_LIBRARY} CACHE INTERNAL "" FORCE)

if(NOT WIN32)
    # 2021-10-07: libzmq's minimum CMake version does not yet support CXX_VISIBILITY_PRESET, so we need to pass the flag manually
    set(cxx_visibility_flag "-fvisibility=hidden")
else()
    set(cxx_visibility_flag "")
endif()

set(lcgpackages http://lcgpackages.web.cern.ch/lcgpackages/tarFiles/sources)
set(libzeromq_version 4.3.5)
ExternalProject_Add(BUILTIN_ZeroMQ
    URL ${lcgpackages}/zeromq-${libzeromq_version}.tar.gz 
    URL_HASH SHA256=6653ef5910f17954861fe72332e68b03ca6e4d9c7160eb3a8de5a5a913bfab43
    CMAKE_ARGS
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCMAKE_INSTALL_LIBDIR=lib
        -DWITH_PERF_TOOL=OFF
        -DBUILD_TESTS=OFF
        -DENABLE_DRAFTS=ON
        -DENABLE_NO_EXPORT=ON
        -DENABLE_CURVE=OFF
        -DWITH_DOCS=OFF
        -DENABLE_WS=OFF
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_CXX_FLAGS=${cxx_visibility_flag}\ ${CMAKE_CXX_FLAGS}\ -w
    BUILD_BYPRODUCTS ${ZeroMQ_LIBRARIES}
    )

set(ZeroMQ_FOUND TRUE CACHE BOOL "" FORCE)

set(ZeroMQ_INCLUDE_DIR  ${ZeroMQ_PREFIX}/include CACHE INTERNAL "" FORCE)
set(ZeroMQ_INCLUDE_DIRS ${ZeroMQ_PREFIX}/include CACHE INTERNAL "" FORCE)

# Workaround to propagate INTERFACE_INCLUDE_DIRECTORIES, see https://stackoverflow.com/a/47358004/1199693
file(MAKE_DIRECTORY ${ZeroMQ_INCLUDE_DIR})

add_library(libzmq INTERFACE IMPORTED GLOBAL)
target_include_directories(libzmq INTERFACE $<BUILD_INTERFACE:${ZeroMQ_INCLUDE_DIR}>)
target_link_libraries(libzmq INTERFACE $<BUILD_INTERFACE:${ZeroMQ_LIBRARIES}>)
add_dependencies(libzmq BUILTIN_ZeroMQ)

set(ZeroMQ_DIR ${ZeroMQ_PREFIX}/share/cmake/ZeroMQ CACHE INTERNAL "" FORCE)
