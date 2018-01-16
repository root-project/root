############################################################################
# CMakeLists.txt file for building ROOT roofitcore package
# @author Pere Mato, CERN
############################################################################

ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-deprecated-declarations) # auto_ptr

if(UNIX AND NOT APPLE)
    # for Linux, BSD, Solaris, Minix: add rt library for timing in RooCPUTimer
    find_library(RT_LIBRARY rt)
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(RooFitCore
        HEADERS "*.h" "MultiProcess/*.h"
        SOURCES "*.cxx" "MultiProcess/*.cxx"
        DICTIONARY_OPTIONS "-writeEmptyRootPCM"
        LIBRARIES ${RT_LIBRARY}
        DEPENDENCIES Core Hist Graf Matrix Tree Minuit Minuit2 RIO MathCore Foam)

if(testing)
    add_subdirectory(test)
endif()
