cmake_minimum_required(VERSION 3.4.3 FATAL_ERROR)
project (cppyy)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_EXTENSIONS OFF)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-strict-aliasing")

message(STATUS "Looking for python interpreter")
#---First look for the python interpreter and fix the version of it for the libraries--
find_package(PythonInterp)

if (PYTHONINTERP_FOUND)
   execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sys;sys.stdout.write(sys.version[:3])"
                   OUTPUT_VARIABLE PYTHON_VERSION)
   execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sys;sys.stdout.write(sys.prefix)"
                   OUTPUT_VARIABLE PYTHON_PREFIX)
   set (CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${PYTHON_PREFIX})
endif()

set(Python_ADDITIONAL_VERSIONS ${PYTHON_VERSION})
find_package(PythonLibs)

if (NOT PYTHONLIBS_FOUND)
   message(FATAL_ERROR "PythonLibs package not found and python component required")
endif()

include_directories ("${PROJECT_SOURCE_DIR}/include" "${PYTHON_INCLUDE_PATH}")

file (GLOB cppyy_src src/*.cxx)

add_library(cppyy SHARED ${cppyy_src})
