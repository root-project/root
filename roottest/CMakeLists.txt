set(ROOT_CONFIG_EXECUTABLE $<TARGET_FILE:root-config>)
if(NOT EXISTS ${ROOT_CONFIG_EXECUTABLE})
  message(FATAL_ERROR "root-config was not found in ${ROOT_CONFIG_EXECUTABLE}.")
endif()
set(ROOT_root_CMD $<TARGET_FILE:root>)
set(ROOT_hadd_CMD $<TARGET_FILE:hadd>)
set(ROOT_genreflex_CMD $<TARGET_FILE:genreflex>)
set(ROOT_rootcint_CMD $<TARGET_FILE:rootcint>)
set(ROOT_rootcling_CMD $<TARGET_FILE:rootcling>)

execute_process(COMMAND ${ROOT_CONFIG_EXECUTABLE} "--tutdir" OUTPUT_VARIABLE ROOT_TUTORIALS_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${ROOT_CONFIG_EXECUTABLE} "--incdir" OUTPUT_VARIABLE ROOT_INCLUDE_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${ROOT_CONFIG_EXECUTABLE} "--incdir" OUTPUT_VARIABLE ROOT_LIBRARY_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${ROOT_CONFIG_EXECUTABLE} "--prefix" OUTPUT_VARIABLE ROOTSYS RESULT_VARIABLE RETVAR OUTPUT_STRIP_TRAILING_WHITESPACE)
# root-config --prefix helps to retrieve ROOTSYS env variable.
if(NOT IS_DIRECTORY ${ROOTSYS})
  message(FATAL_ERROR "ROOTSYS should be set before configuring roottest.")
endif()
  

if(MSVC)
  set(CMAKE_SKIP_TEST_ALL_DEPENDENCY TRUE)
  set(CMAKE_SUPPRESS_REGENERATION TRUE)
  set(libsuffix .dll)
  set(exesuffix .exe)
  set(grep_cmd "findstr")

  set(build_config "--config $<CONFIG>")

  option(win_broken_tests "Enable broken tests on Windows" FALSE)
  option(llvm13_broken_tests "Enable broken tests with LLVM 13 on Windows" FALSE)
  foreach(_opt ${_root_all_options})
    set(ROOT_${_opt} TRUE)
  endforeach()
  # _llvm13_broken_tests only exists with LLVM 13
  if(NOT ROOT_llvm13_broken_tests OR ROOT_llvm13_broken_tests_FOUND)
    set(llvm13_broken_tests TRUE)
  endif()

  if(NOT win_broken_tests)
    set(WILLFAIL_ON_WIN32 WILLFAIL)
  endif()

  cmake_path(CONVERT "${ROOTSYS}" TO_CMAKE_PATH_LIST ROOTSYS)
  set(ROOTSYS ${ROOTSYS} CACHE INTERNAL "")
  cmake_path(CONVERT "${ROOT_TUTORIALS_DIR}" TO_CMAKE_PATH_LIST ROOT_TUTORIALS_DIR)
  cmake_path(CONVERT "${ROOT_INCLUDE_DIRS}" TO_CMAKE_PATH_LIST ROOT_INCLUDE_DIRS)
  cmake_path(CONVERT "${ROOT_LIBRARY_DIR}" TO_CMAKE_PATH_LIST ROOT_LIBRARY_DIR)
  if(CMAKE_GENERATOR MATCHES Ninja)
    set(ROOT_LIBRARIES Core RIO Net Hist Gpad Graf Tree Rint Matrix MathCore)
  else()
    set(ROOT_LIBRARIES libCore libRIO libNet libHist libGpad libGraf libTree libRint libMatrix libMathCore)
  endif()
else()
  # We can not use CMAKE_SHARED_LIBRARY_SUFFIX as on macos it uses the standard dylib rather than the .so
  set(libsuffix .so)
  set(grep_cmd grep)

  # only for non-Windows platforms
  set(OptionalMultiProc MultiProc)

  set(ROOT_LIBRARIES Core RIO Net Hist Gpad Tree Rint Matrix MathCore)
endif()

set(ref_suffix ".ref")

# Detect bitness.
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(64BIT 1)
  if(MSVC)
    set(ref_suffix "_win64.ref")
  endif()
  message("-- Check for bitness: Found 64 bit architecture.")
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
  set(32BIT 1)
  if(MSVC)
    set(ref_suffix "_win32.ref")
  endif()
  message("-- Check for bitness: Found 32 bit architecture.")
else()
  message(FATAL_ERROR "-- Check for bitness: no support for ${CMAKE_SIZEOF_VOID_P}*8 bit processors.")
endif()

# Setup environment.
if (gnuinstall)
  set(ROOTTEST_ENVIRONMENT ROOTIGNOREPREFIX=1)
endif()

# Set some variables that customizes the behaviour of the ROOT macros
set(CMAKE_ROOTTEST_DICT ON)

# Set the CMake module path. Here are all the custom CMake modules.
list(APPEND CMAKE_MODULE_PATH "${ROOT_SOURCE_DIR}/roottest/cmake/modules")

#---Set flag for PyROOT tests that are expected to fail
if(pyroot)
  set(PYTESTS_WILLFAIL WILLFAIL)
endif()

# Find OpenGL
find_library(OPENGL_gl_LIBRARY NAMES GL)

# Set global include dirs and library dirs for roottest.
# These are picked up in the old-style dictionary generation and when standard CMake targets are created.
include_directories(${ROOT_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
link_directories(${ROOT_LIBRARY_DIR})
set(CMAKE_MACOSX_RPATH TRUE)              # use RPATH for MacOSX

include(RoottestCTest)
include(SearchInstalledSoftwareRoottest)

ROOTTEST_COMPILE_MACRO(scripts/utils.cc
                       FIXTURES_SETUP roottest-scripts-utils-fixture)

message("-- Scanning subdirectories for tests...")
ROOTTEST_ADD_TESTDIRS()
