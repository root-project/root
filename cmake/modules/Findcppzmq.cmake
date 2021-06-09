if (NOT (ZeroMQ_FOUND OR TARGET libzmq))
    message(FATAL_ERROR "Search for libzmq first!")
endif()

find_path(cppzmq_INCLUDE_DIRS "zmq.hpp"
        HINTS "${ZeroMQ_INCLUDE_DIR}"
        PATH_SUFFIXES "include" "cppzmq"
        )
mark_as_advanced(cppzmq_INCLUDE_DIRS)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args (cppzmq DEFAULT_MSG cppzmq_INCLUDE_DIRS)

if(cppzmq_FOUND)
    add_library(cppzmq INTERFACE IMPORTED)
    set_target_properties(cppzmq PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${CPPZMQ_INCLUDE_DIRS}"
            INTERFACE_LINK_LIBRARIES libzmq
            )
    set(CPPZMQ_LIBRARIES cppzmq)
endif()