############################################################################
# CMakeLists.txt file for building ROOT math/scipy package
############################################################################
# author: Omar.Zapata@cern.ch 2022

include_directories(SYSTEM ${PYTHON_INCLUDE_DIRS_Development_Main} ${NUMPY_INCLUDE_DIRS})
ROOT_STANDARD_LIBRARY_PACKAGE(Scipy
                              HEADERS 
                                Math/ScipyMinimizer.h
                              LINKDEF 
                                Math/LinkDef.h
                              LIBRARIES ${PYTHON_LIBRARIES_Development_Main}
                              SOURCES
                                src/ScipyMinimizer.cxx
                              DEPENDENCIES Core MathCore RIO
                              )

target_compile_definitions(Scipy PUBLIC USE_ROOT_ERROR ${PYTHON_DEFINITIONS})
target_include_directories(Scipy PUBLIC ${PYTHON_INCLUDE_DIRS})
ROOT_ADD_TEST_SUBDIRECTORY(test)
