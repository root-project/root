############################################################################
# CMakeLists.txt file for building ROOT math/rtools package
# @author Pere Mato, CERN
############################################################################

include_directories(SYSTEM ${R_INCLUDE_DIRS})

ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-overloaded-virtual)

ROOT_STANDARD_LIBRARY_PACKAGE(Rtools
                              HEADERS Math/RMinimizer.h
                              LIBRARIES ${R_LIBRARIES}
                              DEPENDENCIES Core MathCore RInterface)

FILE(COPY "${CMAKE_SOURCE_DIR}/etc/plugins/ROOT@@Math@@Minimizer/P090_RMinimizer.C" DESTINATION "${CMAKE_BINARY_DIR}/etc/plugins/ROOT@@Math@@Minimizer/")

#ROOT_ADD_TEST_SUBDIRECTORY(test)
