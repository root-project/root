# Based on https://github.com/zeromq/cppzmq/blob/a98fa4a91d868a3844e5456741d6782cc1a8d98b/libzmq-pkg-config/FindZeroMQ.cmake
# MIT Licensed:
#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to
#    deal in the Software without restriction, including without limitation the
#    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
#    sell copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:
#
#    The above copyright notice and this permission notice shall be included in
#    all copies or substantial portions of the Software.
#
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
#    IN THE SOFTWARE.

set(PKG_CONFIG_USE_CMAKE_PREFIX_PATH ON)
find_package(PkgConfig)
pkg_check_modules(PC_LIBZMQ QUIET libzmq)

set(ZeroMQ_VERSION ${PC_LIBZMQ_VERSION})

find_path(ZeroMQ_INCLUDE_DIR zmq.h
        PATHS ${ZeroMQ_DIR}/include
        ${PC_LIBZMQ_INCLUDE_DIRS})

find_library(ZeroMQ_LIBRARY
        NAMES zmq
        PATHS ${ZeroMQ_DIR}/lib
        ${PC_LIBZMQ_LIBDIR}
        ${PC_LIBZMQ_LIBRARY_DIRS})

set ( ZeroMQ_LIBRARIES ${ZeroMQ_LIBRARY} )
set ( ZeroMQ_INCLUDE_DIRS ${ZeroMQ_INCLUDE_DIR} )

# check for zmq_ppoll
if(ZeroMQ_FOUND)
    SET(SAVE_CMAKE_REQUIRED_DEFINITIONS "${CMAKE_REQUIRED_DEFINITIONS}")
    SET(CMAKE_REQUIRED_DEFINITIONS "-DZMQ_BUILD_DRAFT_API")
    check_cxx_symbol_exists(zmq_ppoll zmq.h ZeroMQ_HAS_PPOLL)
    SET(CMAKE_REQUIRED_DEFINITIONS "${SAVE_CMAKE_REQUIRED_DEFINITIONS}")
endif()

include ( FindPackageHandleStandardArgs )
# handle the QUIETLY and REQUIRED arguments and set ZeroMQ_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args ( ZeroMQ DEFAULT_MSG ZeroMQ_LIBRARIES ZeroMQ_INCLUDE_DIRS ZeroMQ_HAS_PPOLL)

if(ZeroMQ_FOUND)
    if(NOT TARGET libzmq)
        add_library(libzmq UNKNOWN IMPORTED)
        set_target_properties(libzmq PROPERTIES
                IMPORTED_LOCATION ${ZeroMQ_LIBRARIES}
                INTERFACE_INCLUDE_DIRECTORIES ${ZeroMQ_INCLUDE_DIRS})
    endif()
endif()