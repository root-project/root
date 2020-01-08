find_package(ZeroMQ REQUIRED)

ROOT_ADD_GTEST(testRooFitZMQ ZMQ.cpp LIBRARIES
        RooFitZMQ ${ZeroMQ_LIBRARY}
        RooFitMultiProcess # wait_for_child in util
)
