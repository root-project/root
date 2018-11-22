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
    SOURCES
        src/ZeroMQSvc.cpp
        src/ZeroMQPoller.cpp
        src/functions.cpp
    DICTIONARY_OPTIONS
        -I${ZeroMQ_INCLUDE_DIR}
        -writeEmptyRootPCM
    DEPENDENCIES
        Core # TVersionCheck, RegisterModule
        RIO  # TBufferFile, TBufferIO
    LIBRARIES
        ${ZeroMQ_LIBRARY}
    )

target_include_directories(RooFitZMQ PUBLIC ${ZeroMQ_INCLUDE_DIR})

if(testing)
    ROOT_ADD_TEST_SUBDIRECTORY(test)
endif()
