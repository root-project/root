find_package(ROOT REQUIRED CONFIG)

if(MSVC)
  set(CMAKE_SKIP_TEST_ALL_DEPENDENCY TRUE)
  set(CMAKE_SUPPRESS_REGENERATION TRUE)
  set(libsuffix .dll)
  set(exesuffix .exe)
  set(CMAKE_CXX_FLAGS "/nologo /Zc:__cplusplus /MD /GR /EHsc- /W3 /D_WIN32")
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

  set(CMAKE_CXX_FLAGS ${ROOT_CXX_FLAGS})
  # ROOT_BINDIR is set by ROOTConfig.cmake
  set(ROOT_CONFIG_EXECUTABLE ${ROOT_BINDIR}/root-config.bat)
  execute_process(COMMAND ${ROOT_CONFIG_EXECUTABLE} "--prefix" OUTPUT_VARIABLE ROOTSYS RESULT_VARIABLE RETVAR OUTPUT_STRIP_TRAILING_WHITESPACE)
  cmake_path(CONVERT "${ROOTSYS}" TO_CMAKE_PATH_LIST ROOTSYS)
  set(ROOTSYS ${ROOTSYS} CACHE INTERNAL "")
  execute_process(COMMAND ${ROOT_CONFIG_EXECUTABLE} "--tutdir" OUTPUT_VARIABLE ROOT_TUTORIALS_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  cmake_path(CONVERT "${ROOT_TUTORIALS_DIR}" TO_CMAKE_PATH_LIST ROOT_TUTORIALS_DIR)
  set(ROOT_root_CMD ${ROOTSYS}/bin/root.exe)
  set(ROOT_hadd_CMD ${ROOTSYS}/bin/hadd.exe)
  set(ROOT_genreflex_CMD ${ROOTSYS}/bin/genreflex.exe)
  set(ROOT_rootcint_CMD ${ROOTSYS}/bin/rootcint.exe)
  set(ROOT_rootcling_CMD ${ROOTSYS}/bin/rootcling.exe)
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

  # root-config --prefix helps to retrieve ROOTSYS env variable.
  # ROOT_BINDIR is set by ROOTConfig.cmake
  find_program(ROOT_CONFIG_EXECUTABLE root-config PATHS ${ROOT_BINDIR})
  if(NOT ROOT_CONFIG_EXECUTABLE)
    message(ERROR "root-config is not found, please rerun configuration step.")
  endif()
  execute_process(COMMAND ${ROOT_CONFIG_EXECUTABLE} "--prefix" OUTPUT_VARIABLE ROOTSYS RESULT_VARIABLE RETVAR OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(ROOTSYS ${ROOTSYS} CACHE INTERNAL "")
  execute_process(COMMAND ${ROOT_CONFIG_EXECUTABLE} "--tutdir" OUTPUT_VARIABLE ROOT_TUTORIALS_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(ROOT_LIBRARIES Core RIO Net Hist Gpad Tree Rint Matrix MathCore)
  set(ROOT_root_CMD ${ROOTSYS}/bin/root.exe)
  set(ROOT_hadd_CMD ${ROOTSYS}/bin/hadd)
  set(ROOT_genreflex_CMD ${ROOTSYS}/bin/genreflex)
  set(ROOT_rootcint_CMD ${ROOTSYS}/bin/rootcint)
  set(ROOT_rootcling_CMD rootcling)
endif()

get_filename_component(ROOT_LIBRARY_DIR "${ROOTSYS}/lib" ABSOLUTE)

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
set(ROOTTEST_ENV_PATH ${ROOT_BINDIR})
if(MSVC)
  set(ROOTTEST_ENV_PYTHONPATH ${ROOT_BINDIR})
else()
  set(ROOTTEST_ENV_PYTHONPATH ${ROOT_LIBRARY_DIR})
endif()
set(ROOTTEST_ENV_EXTRA)

if(MSVC)
  set(ROOTTEST_ENVIRONMENT
      PYTHONPATH=${ROOTTEST_ENV_PYTHONPATH})
else()
  # `PYTHONPATH=...:` (trailing `:`) kills roottest/python/stl/PyROOT_stltests.py.
  string(REGEX REPLACE ":$" "" TESTPYTHONPATH "${ROOTTEST_ENV_PYTHONPATH}:$ENV{PYTHONPATH}")
  set(ROOTTEST_ENVIRONMENT
      PATH=${ROOTTEST_ENV_PATH}:$ENV{PATH}
      PYTHONPATH=${TESTPYTHONPATH})
  if (gnuinstall)
    set(ROOTTEST_ENVIRONMENT ${ROOTTEST_ENVIRONMENT} ROOTIGNOREPREFIX=1)
  endif()
endif()

# Set some variables that customizes the behaviour of the ROOT macros
set(CMAKE_ROOTTEST_DICT ON)

# Set the CMake module path. Here are all the custom CMake modules.
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${ROOT_SOURCE_DIR}/roottest/cmake/modules")

#---Set flag for PyROOT tests that are expected to fail
if(pyroot)
  set(PYTESTS_WILLFAIL WILLFAIL)
endif()

# Find OpenGL
find_library(OPENGL_gl_LIBRARY NAMES GL)

# Setup standard includes / links.
include_directories(${ROOT_INCLUDE_DIRS} ${ROOT_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
link_directories(${ROOT_LIBRARY_DIR})
set(CMAKE_MACOSX_RPATH TRUE)              # use RPATH for MacOSX

include(RoottestCTest)
include(SearchInstalledSoftwareRoottest)

ROOTTEST_COMPILE_MACRO(scripts/utils.cc
                       FIXTURES_SETUP roottest-scripts-utils-fixture)

message("-- Scanning subdirectories for tests...")
ROOTTEST_ADD_TESTDIRS()
