cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(roottest)

# If no explicit ROOTSYS is set, use ${ROOTSYS} environment variable to search
# for ROOT. This may be set by sourcing thisroot.sh from an installation.
if(NOT DEFINED ROOTSYS AND DEFINED ENV{ROOTSYS})
  file(TO_CMAKE_PATH $ENV{ROOTSYS} ROOTSYS)
  list(INSERT CMAKE_PREFIX_PATH 0 ${ROOTSYS})
endif()

# Use ${ROOTSYS} CMake variable to search for ROOT. This is set when roottest is
# built as part of ROOT, i.e., when ROOT is configured with -Droottest=ON. It
# may also be set by passing -DROOTSYS=<PATH> to CMake when configuring a
# standalone roottest against an installation of ROOT. If this is set, it takes
# precedence over the environment to ensure that when roottest is being built in
# a ROOT build configured with -Droottest=ON, it will find the ROOT being built,
# and not a ROOT installation sourced with thisroot.sh or found in $PATH.
if(DEFINED ROOTSYS)
  list(INSERT CMAKE_PREFIX_PATH 0 ${ROOTSYS})
endif()

find_package(ROOT REQUIRED CONFIG)

if(NOT DEFINED PYTHON_EXECUTABLE)
  find_package(PythonInterp ${ROOT_PYTHON_VERSION} REQUIRED QUIET)
endif()

if(MSVC)

  set(libsuffix .dll)
  set(exesuffix .exe)
  set(CMAKE_CXX_FLAGS "/nologo /Zc:__cplusplus /MD /GR /EHsc- /W3 /D_WIN32")
  set(grep_cmd "findstr")

  option(win_broken_tests "Enable broken tests on Windows" FALSE)

  if(NOT win_broken_tests)
    set(WILLFAIL_ON_WIN32 WILLFAIL)
  endif()
  if (NOT PYTHON_EXECUTABLE)
    find_package(PythonInterp)
  endif()

  if(DEFINED ROOT_SOURCE_DIR)
    # ROOT_BINDIR is set by ROOTConfig.cmake
    get_filename_component(realfp "${ROOT_BINDIR}/../" ABSOLUTE)
    file(TO_CMAKE_PATH ${realfp} ROOTSYS)
    set(ROOT_LIBRARIES libCore libRIO libNet libHist libGpad libTree libRint libMatrix libMathCore)
  endif()

  include(${ROOT_USE_FILE})
  set(ROOTSYS ${ROOTSYS} CACHE INTERNAL "")
  file(READ "${ROOTSYS}/bin/root-config" root_config)
  string(REGEX MATCH "LZ4_VERSION=([0-9]+)\\.([0-9]+)\\.([0-9]+)" LZ4_VERSION_STR ${root_config})
  string(REGEX MATCH "ZSTD_VERSION=*([0-9]+)\\.([0-9]+)\\.([0-9]+)" ZSTD_VERSION_STR ${root_config})
  string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)" LZ4_VERSION ${LZ4_VERSION_STR})
  string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)" ZSTD_VERSION ${ZSTD_VERSION_STR})
  if(NOT LZ4_VERSION)
    message(FATAL_ERROR "LZ4_VERSION not found in ${ROOTSYS}/bin/root-config")
  endif()
  if(NOT ZSTD_VERSION)
    message(FATAL_ERROR "ZSTD_VERSION not found in ${ROOTSYS}/bin/root-config")
  endif()
  set(ROOT_root_CMD ${ROOTSYS}/bin/root.exe)
  set(ROOT_hadd_CMD ${ROOTSYS}/bin/hadd.exe)
  set(ROOT_genreflex_CMD ${ROOTSYS}/bin/genreflex.exe)
  set(ROOT_rootcint_CMD ${ROOTSYS}/bin/rootcint.exe)
  set(ROOT_rootcling_CMD ${ROOTSYS}/bin/rootcling.exe)
  enable_testing()

else()

  # We can not use CMAKE_SHARED_LIBRARY_SUFFIX as on macos it uses the standard dylib rather than the .so
  set(libsuffix .so)
  set(grep_cmd grep)

  # only for non-Windows platforms
  set(OptionalMultiProc MultiProc)

  # If ROOT_SOURCE_DIR is set, roottest is built as part of the root build.
  # If ROOT_SOURCE_DIR is not set, search for an installation.
  if(DEFINED ROOT_SOURCE_DIR)
    # root-config --prefix helps to retrieve ROOTSYS env variable.
    # ROOT_BINDIR is set by ROOTConfig.cmake
    find_program(ROOT_CONFIG_EXECUTABLE root-config PATHS ${ROOT_BINDIR})
    if(NOT ROOT_CONFIG_EXECUTABLE)
      message(ERROR "root-config is not found, please rerun configuration step.")
    endif()
    exec_program(${ROOT_CONFIG_EXECUTABLE} ARGS "--prefix" OUTPUT_VARIABLE ROOTSYS RETURN_VALUE RETVAR)
    set(ROOTSYS ${ROOTSYS} CACHE INTERNAL "")
    set(ROOT_LIBRARIES Core RIO Net Hist Gpad Tree Rint Matrix MathCore)
    set(ROOT_root_CMD ${ROOTSYS}/bin/root.exe)
    set(ROOT_hadd_CMD ${ROOTSYS}/bin/hadd)
    set(ROOT_genreflex_CMD ${ROOTSYS}/bin/genreflex)
    set(ROOT_rootcint_CMD ${ROOTSYS}/bin/rootcint)
    set(ROOT_rootcling_CMD rootcling)
  else() # standalone roottest.git
    include(${ROOT_USE_FILE})
    # root-config --prefix helps to retrieve ROOTSYS env variable.
    find_program(ROOT_CONFIG_EXECUTABLE root-config)
    exec_program(${ROOT_CONFIG_EXECUTABLE} ARGS "--prefix" OUTPUT_VARIABLE ROOTSYS RETURN_VALUE RETVAR)
    set(ROOTSYS ${ROOTSYS} CACHE INTERNAL "")
    # Some roottest.git tests are checking LZ4 and ZSTD, which has different references depends on its version.
    # We recover LZ4/ZSTD version used in ROOT, stored in root-config --config output.
    exec_program(${ROOT_CONFIG_EXECUTABLE} ARGS "--srcdir" OUTPUT_VARIABLE ROOT_SOURCE_DIR)
    exec_program(${ROOT_CONFIG_EXECUTABLE} ARGS "--config" OUTPUT_VARIABLE ROOT_CONFIG_VARIABLES RETURN_VALUE RETVAR)
    separate_arguments(ROOT_CONFIG_VARIABLES)
    foreach(item ${ROOT_CONFIG_VARIABLES})
      if("${item}" MATCHES "LZ4_VERSION=(.*)")
        string(REGEX REPLACE "LZ4_VERSION=(.*)" "\\1" LZ4 ${item})
        set(LZ4_VERSION ${LZ4} CACHE INTERNAL "")
      elseif("${item}" MATCHES "ZSTD_VERSION=(.*)")
        string(REGEX REPLACE "ZSTD_VERSION=(.*)" "\\1" ZSTD ${item})
        set(ZSTD_VERSION ${ZSTD} CACHE INTERNAL "")
      endif()
    endforeach()
    # If we didn't manage to find LZ4_VERSION/ZSTD_VERSION in "root-config --config" output,
    # we consider that ROOT had used system LZ4/ZSTD and we can find it via find_package().
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake/modules ${ROOTSYS}/cmake/modules ${ROOT_SOURCE_DIR}/cmake/modules)
    if(NOT LZ4_VERSION)
      find_package(LZ4)
      if(NOT LZ4_FOUND)
        message(FATAL_ERROR "lz4 is required to be installed before running roottest.git")
      endif()
    endif()
    if(NOT ZSTD_VERSION)
      find_package(ZSTD)
      if(NOT ZSTD_FOUND)
        message(FATAL_ERROR "zstd is required to be installed before running roottest.git")
      endif()
    endif()
    # To be able to use the same C++ standard as used in ROOT, we can try to retrieve it from root-config --features
    exec_program(root-config ARGS "--features" OUTPUT_VARIABLE ROOT_FEATURES RETURN_VALUE RETVAR)
    separate_arguments(ROOT_FEATURES)
    foreach(item ${ROOT_FEATURES})
      if("${item}" MATCHES "^cxx(.*)")
        string(REGEX REPLACE "^cxx(.*)" "\\1" CXX_VERSION ${item})
        set(CMAKE_CXX_STANDARD ${CXX_VERSION} CACHE INTERNAL "")
      endif()
    endforeach()
    find_program(ROOT_root_CMD root.exe PATHS ${ROOTSYS}/bin)
    find_program(ROOT_hadd_CMD hadd PATHS ${ROOTSYS}/bin)
    find_program(ROOT_genreflex_CMD genreflex PATHS ${ROOTSYS}/bin)
    find_program(ROOT_rootcint_CMD rootcint PATHS ${ROOTSYS}/bin)
    find_program(ROOT_rootcling_CMD rootcling PATHS ${ROOTSYS}/bin)
    enable_testing()
  endif()
endif()

# Synchronizing default compression algorithm used for ROOT.
# We need to have it for CMake settings for switching tests references.
# FIXME: it should be exported from ROOT build.
set(compression_default "zlib" CACHE STRING "" FORCE)

get_filename_component(ROOT_LIBRARY_DIR "${ROOTSYS}/lib" ABSOLUTE)

# Detect bitness.
if(CMAKE_SYSTEM_PROCESSOR MATCHES amd64.*|x86_64.*|aarch64.*|ppc64.*
   OR (CMAKE_VERSION VERSION_LESS 3.0 AND CMAKE_SYSTEM_NAME STREQUAL Darwin) )
  set(64BIT 1)
  message("-- Check for bitness: Found 64 bit architecture.")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES i686.*|i386.*|x86.*)
  set(32BIT 1)
  message("-- Check for bitness: Found 32 bit architecture.")
endif()

if (MSVC)
  if("${CMAKE_GENERATOR}" MATCHES "(Win64|IA64)")
    set(64BIT 1)
    set(ref_suffix "_win64.ref")
    message(STATUS "-- Check for generator architecture: Found 64 bit architecture.")
  else()
    set(32BIT 1)
    set(ref_suffix "_win32.ref")
    message(STATUS "-- Check for generator architecture: Found 32 bit architecture.")
  endif()
else()
  set(ref_suffix ".ref")
endif()

# Setup environment.
set(ROOTTEST_ENV_PATH ${ROOT_BINDIR})
if(MSVC)
  set(ROOTTEST_ENV_PYTHONPATH ${ROOT_BINDIR})
else()
  set(ROOTTEST_ENV_PYTHONPATH ${ROOT_LIBRARY_DIR})
endif()
set(ROOTTEST_ENV_LIBRARYPATH ${ROOT_LIBRARY_DIR})
set(ROOTTEST_ENV_EXTRA)

if(MSVC)
  set(ROOTTEST_ENVIRONMENT
      ROOTSYS=${ROOTSYS}
      PYTHONPATH=${ROOTTEST_ENV_PYTHONPATH})
else()
  set(ROOTTEST_ENVIRONMENT
      ROOTSYS=${ROOTSYS}
      PATH=${ROOTTEST_ENV_PATH}:$ENV{PATH}
      PYTHONPATH=${ROOTTEST_ENV_PYTHONPATH}:$ENV{PYTHONPATH}
      ${ld_library_path}=${ROOTTEST_ENV_LIBRARYPATH}:$ENV{${ld_library_path}})
  if (gnuinstall)
    set(ROOTTEST_ENVIRONMENT ${ROOTTEST_ENVIRONMENT} ROOTIGNOREPREFIX=1)
  endif()
endif()

# Resolve symbolic links for the ROOTTEST_DIR variable.
get_filename_component(ROOTTEST_DIR ${CMAKE_CURRENT_SOURCE_DIR} ABSOLUTE)

# Set some variables that customizes the behaviour of the ROOT macros
set(CMAKE_ROOTTEST_DICT ON)

# Set the CMake module path. Here are all the custom CMake modules.
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${ROOTTEST_DIR}/cmake/modules")

# Find python.
if(ROOT_pyroot_FOUND)
  if (NOT PYTHON_EXECUTABLE)
    find_package(PythonInterp)
    if(PYTHONINTERP_FOUND)
      execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sys;sys.stdout.write(sys.prefix)"
                       OUTPUT_VARIABLE PYTHON_PREFIX)
      set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${PYTHON_PREFIX})
    endif()
    if (NOT PYTHON_LIBRARIES)
      find_package(PythonLibs)
    endif()
  endif()
endif()

#---Set flag for PyROOT tests that are expected to fail
if(ROOT_pyroot_FOUND)
  set(PYTESTS_WILLFAIL WILLFAIL)
endif()

# Find OpenGL
find_library(OPENGL_gl_LIBRARY NAMES GL)

# Setup standard includes / links.
include_directories(${ROOT_INCLUDE_DIRS} ${ROOT_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
link_directories(${ROOT_LIBRARY_DIR})
set(CMAKE_MACOSX_RPATH TRUE)              # use RPATH for MacOSX

include(RoottestCTest)
include(RoottestMacros)
include(SearchInstalledSoftwareRoottest)

message("-- Scanning subdirectories for tests...")
ROOTTEST_ADD_TESTDIRS()
