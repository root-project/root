if (NOT (ZeroMQ_FOUND OR TARGET libzmq))
    message(FATAL_ERROR "Search for libzmq first!")
endif()

find_path(cppzmq_INCLUDE_DIRS "zmq.hpp"
        HINTS "${ZeroMQ_INCLUDE_DIR}"
        PATH_SUFFIXES "include" "cppzmq"
        )
mark_as_advanced(cppzmq_INCLUDE_DIRS)

# check for required minimum version for RooFitZMQ
if(cppzmq_INCLUDE_DIRS)
    SET(SAVE_CMAKE_REQUIRED_INCLUDES "${CMAKE_REQUIRED_INCLUDES}")
    SET(CMAKE_REQUIRED_INCLUDES ${cppzmq_INCLUDE_DIRS} ${ZeroMQ_INCLUDE_DIRS})
    SET(SAVE_CMAKE_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES}")
    SET(CMAKE_REQUIRED_LIBRARIES libzmq)
    check_cxx_source_runs("#include <zmq.hpp> \n int main () {if (CPPZMQ_VERSION_MAJOR > 4 || (CPPZMQ_VERSION_MAJOR == 4 && CPPZMQ_VERSION_MINOR >= 8)) {return 0;} else {return 1;} }" cppzmq_VERSION_COMPATIBLE)
    SET(CMAKE_REQUIRED_INCLUDES "${SAVE_CMAKE_REQUIRED_INCLUDES}")
    SET(CMAKE_REQUIRED_LIBRARIES "${SAVE_CMAKE_REQUIRED_LIBRARIES}")

    if (NOT QUIET AND NOT cppzmq_VERSION_COMPATIBLE)
        message("-- Version of zmq.hpp at ${cppzmq_INCLUDE_DIRS} too old, need at least 4.8.0.")
    endif()
endif()

include(FindPackageHandleStandardArgs)
# handle the QUIET and REQUIRED arguments and set cppzmq_FOUND to TRUE
# if all listed variables are truthy
find_package_handle_standard_args (cppzmq DEFAULT_MSG cppzmq_INCLUDE_DIRS cppzmq_VERSION_COMPATIBLE)

if(cppzmq_FOUND)
    add_library(cppzmq INTERFACE IMPORTED)
    set_target_properties(cppzmq PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${CPPZMQ_INCLUDE_DIRS}"
            INTERFACE_LINK_LIBRARIES libzmq
            )
    set(CPPZMQ_LIBRARIES cppzmq)
endif()