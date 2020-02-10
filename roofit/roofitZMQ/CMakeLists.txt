############################################################################
# CMakeLists.txt file for building ROOT roofitcore/ZMQ package
# @author Patrick Bos, NL eScience Center
############################################################################

find_package(ZeroMQ REQUIRED)

ROOT_STANDARD_LIBRARY_PACKAGE(RooFitZMQ
    HEADERS
        RooFit_ZMQ/zmq.hxx
        RooFit_ZMQ/Utility.h
        RooFit_ZMQ/ZeroMQSvc.h
        RooFit_ZMQ/functions.h
        RooFit_ZMQ/ZeroMQPoller.h
        RooFit_ZMQ/ppoll.h
    SOURCES
        src/ZeroMQSvc.cpp
        src/ZeroMQPoller.cpp
        src/functions.cpp
        src/ppoll.cpp
        src/libzmq/clock.cpp
        src/libzmq/err.cpp
        src/libzmq/polling_util.cpp
    DICTIONARY_OPTIONS
        -I${ZeroMQ_INCLUDE_DIR}
        -writeEmptyRootPCM
    DEPENDENCIES
        Core # TVersionCheck, RegisterModule, these are always called from G__RooFitZMQ.cxx.o
#        RIO  # TBufferFile, TBufferIO
    LIBRARIES
        ${ZeroMQ_LIBRARY}
    )

target_include_directories(RooFitZMQ PUBLIC ${ZeroMQ_INCLUDE_DIR})

if(testing)
    ROOT_ADD_TEST_SUBDIRECTORY(test)
endif()
