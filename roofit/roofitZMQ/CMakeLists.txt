############################################################################
# CMakeLists.txt file for building ROOT roofitcore/ZMQ package
# @author Patrick Bos, Netherlands eScience Center
############################################################################

ROOT_LINKER_LIBRARY(RooFitZMQ
        src/ZeroMQSvc.cpp
        src/ZeroMQPoller.cpp
        src/functions.cpp
        src/ppoll.cpp
    )

target_link_libraries(RooFitZMQ PUBLIC libzmq cppzmq)
set(RooFitZMQ_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/res)
target_include_directories(RooFitZMQ
        PRIVATE ${RooFitZMQ_INCLUDE_DIR}
        INTERFACE $<BUILD_INTERFACE:${RooFitZMQ_INCLUDE_DIR}>)

# zmq_ppoll is still in the draft API, and RooFitZMQ relies on it
target_compile_definitions(RooFitZMQ PUBLIC ZMQ_BUILD_DRAFT_API)
# to avoid leaking symbols
target_compile_definitions(RooFitZMQ PUBLIC ZMQ_NO_EXPORT)

ROOT_ADD_TEST_SUBDIRECTORY(test)
