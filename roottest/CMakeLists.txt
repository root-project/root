cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

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

if(NOT DEFINED Python3_EXECUTABLE)
  find_package(Python3 ${ROOT_PYTHON_VERSION} REQUIRED QUIET COMPONENTS Interpreter)
endif()

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
  if (NOT Python3_EXECUTABLE)
    find_package(PythonInterp)
  endif()

  if(DEFINED ROOT_SOURCE_DIR)
    set(CMAKE_CXX_FLAGS ${ROOT_CXX_FLAGS})
    # ROOT_BINDIR is set by ROOTConfig.cmake
    set(ROOT_CONFIG_EXECUTABLE ${ROOT_BINDIR}/root-config.bat)
    execute_process(COMMAND ${ROOT_CONFIG_EXECUTABLE} "--prefix" OUTPUT_VARIABLE ROOTSYS RESULT_VARIABLE RETVAR OUTPUT_STRIP_TRAILING_WHITESPACE)
    file(TO_CMAKE_PATH ${ROOTSYS} ROOTSYS)
    set(ROOTSYS ${ROOTSYS} CACHE INTERNAL "")
    execute_process(COMMAND ${ROOT_CONFIG_EXECUTABLE} "--tutdir" OUTPUT_VARIABLE ROOT_TUTORIALS_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    file(TO_CMAKE_PATH ${ROOT_TUTORIALS_DIR} ROOT_TUTORIALS_DIR)
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
    include(${ROOT_USE_FILE})
    set(ROOT_CONFIG_EXECUTABLE root-config.bat)
    execute_process(COMMAND ${ROOT_CONFIG_EXECUTABLE} "--prefix" OUTPUT_VARIABLE ROOTSYS RESULT_VARIABLE RETVAR OUTPUT_STRIP_TRAILING_WHITESPACE)
    file(TO_CMAKE_PATH ${ROOTSYS} ROOTSYS)
    set(ROOTSYS ${ROOTSYS} CACHE INTERNAL "")
    execute_process(COMMAND ${ROOT_CONFIG_EXECUTABLE} "--srcdir" OUTPUT_VARIABLE ROOT_SOURCE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    file(TO_CMAKE_PATH ${ROOT_SOURCE_DIR} ROOT_SOURCE_DIR)
    execute_process(COMMAND ${ROOT_CONFIG_EXECUTABLE} "--tutdir" OUTPUT_VARIABLE ROOT_TUTORIALS_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    file(TO_CMAKE_PATH ${ROOT_TUTORIALS_DIR} ROOT_TUTORIALS_DIR)
    execute_process(COMMAND ${ROOT_CONFIG_EXECUTABLE} "--features" OUTPUT_VARIABLE ROOT_FEATURES RESULT_VARIABLE RETVAR OUTPUT_STRIP_TRAILING_WHITESPACE)
    separate_arguments(ROOT_FEATURES)
    foreach(item ${ROOT_FEATURES})
      if("${item}" MATCHES "^cxx(.*)")
        string(REGEX REPLACE "^cxx(.*)" "\\1" CXX_VERSION ${item})
        set(CMAKE_CXX_STANDARD ${CXX_VERSION} CACHE INTERNAL "")
      endif()
    endforeach()
    # Some roottest.git tests are checking LZ4 and ZSTD, which has different references depends on its version.
    # We recover LZ4/ZSTD version used in ROOT, stored in root-config --config output.
    execute_process(COMMAND ${ROOT_CONFIG_EXECUTABLE} "--config" OUTPUT_VARIABLE ROOT_CONFIG_VARIABLES RESULT_VARIABLE RETVAR OUTPUT_STRIP_TRAILING_WHITESPACE)
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
    if(NOT LZ4_VERSION)
      message(FATAL_ERROR "LZ4_VERSION not found in ${ROOTSYS}/bin/root-config.bat")
    endif()
    if(NOT ZSTD_VERSION)
      message(FATAL_ERROR "ZSTD_VERSION not found in ${ROOTSYS}/bin/root-config.bat")
    endif()
    find_program(ROOT_root_CMD root.exe PATHS ${ROOTSYS}/bin)
    find_program(ROOT_hadd_CMD hadd.exe PATHS ${ROOTSYS}/bin)
    find_program(ROOT_genreflex_CMD genreflex.exe PATHS ${ROOTSYS}/bin)
    find_program(ROOT_rootcint_CMD rootcint.exe PATHS ${ROOTSYS}/bin)
    find_program(ROOT_rootcling_CMD rootcling.exe PATHS ${ROOTSYS}/bin)
    enable_testing()
  endif()
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
    execute_process(COMMAND ${ROOT_CONFIG_EXECUTABLE} "--prefix" OUTPUT_VARIABLE ROOTSYS RESULT_VARIABLE RETVAR OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(ROOTSYS ${ROOTSYS} CACHE INTERNAL "")
    execute_process(COMMAND ${ROOT_CONFIG_EXECUTABLE} "--tutdir" OUTPUT_VARIABLE ROOT_TUTORIALS_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
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
    execute_process(COMMAND ${ROOT_CONFIG_EXECUTABLE} "--prefix" OUTPUT_VARIABLE ROOTSYS RESULT_VARIABLE RETVAR OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(ROOTSYS ${ROOTSYS} CACHE INTERNAL "")
    # Some roottest.git tests are checking LZ4 and ZSTD, which has different references depends on its version.
    # We recover LZ4/ZSTD version used in ROOT, stored in root-config --config output.
    execute_process(COMMAND ${ROOT_CONFIG_EXECUTABLE} "--srcdir" OUTPUT_VARIABLE ROOT_SOURCE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${ROOT_CONFIG_EXECUTABLE} "--tutdir" OUTPUT_VARIABLE ROOT_TUTORIALS_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${ROOT_CONFIG_EXECUTABLE} "--config" OUTPUT_VARIABLE ROOT_CONFIG_VARIABLES RESULT_VARIABLE RETVAR OUTPUT_STRIP_TRAILING_WHITESPACE)
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
    execute_process(COMMAND ${ROOT_CONFIG_EXECUTABLE} "--features" OUTPUT_VARIABLE ROOT_FEATURES RESULT_VARIABLE RETVAR OUTPUT_STRIP_TRAILING_WHITESPACE)
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
set(ROOTTEST_ENV_LIBRARYPATH ${ROOT_LIBRARY_DIR})
set(ROOTTEST_ENV_EXTRA)

# Optionally, add Fortran link directories. This is needed in particular for
# macOS, where the path of the Fortran library might not be in the default
# DYLD_LIBRARY_PATH, but it gets picked up with enable_language(Fortran).
if(APPLE AND fortran)
  # Convert Fortran link dirs to colon-separated string
  string(REPLACE ";" ":" FORTRAN_LINK_PATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}")

  # Append to ROOTTEST_ENV_LIBRARYPATH
  set(ROOTTEST_ENV_LIBRARYPATH "${ROOTTEST_ENV_LIBRARYPATH}:${FORTRAN_LINK_PATH}")

  # Deduplicate ROOTTEST_ENV_LIBRARYPATH
  string(REPLACE ":" ";" _tmp "${ROOTTEST_ENV_LIBRARYPATH}")
  list(REMOVE_DUPLICATES _tmp)
  string(REPLACE ";" ":" ROOTTEST_ENV_LIBRARYPATH "${_tmp}")
endif()

if(MSVC)
  set(ROOTTEST_ENVIRONMENT
      ROOTSYS=${ROOTSYS}
      PYTHONPATH=${ROOTTEST_ENV_PYTHONPATH})
else()
  # `PYTHONPATH=...:` (trailing `:`) kills roottest/python/stl/PyROOT_stltests.py.
  string(REGEX REPLACE ":$" "" TESTPYTHONPATH "${ROOTTEST_ENV_PYTHONPATH}:$ENV{PYTHONPATH}")
  set(ROOTTEST_ENVIRONMENT
      ROOTSYS=${ROOTSYS}
      PATH=${ROOTTEST_ENV_PATH}:$ENV{PATH}
      PYTHONPATH=${TESTPYTHONPATH}
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
  if (NOT Python3_EXECUTABLE)
    find_package(PythonInterp)
    if(PYTHONINTERP_FOUND)
      execute_process(COMMAND ${Python3_EXECUTABLE} -c "import sys;sys.stdout.write(sys.prefix)"
                       OUTPUT_VARIABLE PYTHON_PREFIX)
      set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${PYTHON_PREFIX})
    endif()
    if (NOT Python3_LIBRARIES)
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
