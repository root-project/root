# Library which powers fast batch computations in Roofit.
#
# This file is used in two different ways:
#
#   1. It is pulled in with add_subdirectory() by the regular ROOT build. It
#      then defines libRooBatchCompute and all its backends: the CPU ones
#      always, and the CUDA one if ROOT was configured with -Dcuda=ON.
#
#   2. It can be used directly as the top-level CMakeLists.txt of a standalone
#      project that builds *only* the CUDA backend, libRooBatchCompute_CUDA,
#      against an already installed ROOT. This makes it possible to distribute a
#      CUDA-free ROOT and to ship the GPU backend as a separate package, without
#      having to maintain a copy of this file downstream. See README.md in this
#      directory for the exact invocation.
#
# The two modes share the same source lists and the same target configuration,
# so there is only one place where the list of files is maintained.

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)

  cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

  project(RooBatchComputeCUDA LANGUAGES CXX CUDA)

  set(RooBatchCompute_STANDALONE_CUDA TRUE)

  # The standalone project exists to build the CUDA backend, nothing else.
  set(cuda ON)

  # The ROOT build defaults to an optimized build, but we deliberately don't
  # pick a build type here: leaving CMAKE_BUILD_TYPE empty and passing the
  # optimization flags in CMAKE_CXX_FLAGS and CMAKE_CUDA_FLAGS instead is a
  # common packaging idiom, and forcing the cache entry would silently add a
  # second set of flags on top of those. Only warn about the case where neither
  # of the two is given, because the kernels then end up unoptimized.
  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES
     AND NOT "${CMAKE_CUDA_FLAGS} ${CMAKE_CXX_FLAGS}" MATCHES "(^| )-O")
    message(WARNING
        "No CMAKE_BUILD_TYPE was specified, so the CUDA kernels will be compiled "
        "without any optimization flags. Configure with -DCMAKE_BUILD_TYPE=Release "
        "for an optimized build like the one the ROOT build does by default, or "
        "pass your own optimization flags in CMAKE_CXX_FLAGS and CMAKE_CUDA_FLAGS.")
  endif()

  find_package(ROOT REQUIRED CONFIG)

  if(NOT TARGET ROOT::RooBatchCompute)
    message(FATAL_ERROR
        "The ROOT installation in ${ROOT_DIR} does not provide libRooBatchCompute. "
        "It was most likely configured with -Droofit=OFF, in which case the CUDA "
        "backend is of no use. Point CMAKE_PREFIX_PATH or ROOT_DIR at a ROOT "
        "installation that includes RooFit.")
  endif()

  if(TARGET ROOT::RooBatchCompute_CUDA)
    message(WARNING
        "The ROOT installation in ${ROOT_DIR} was configured with -Dcuda=ON and "
        "already provides libRooBatchCompute_CUDA. Installing this project with the "
        "default RooBatchCompute_CUDA_INSTALL_LIBDIR will overwrite it.")
  endif()

  # Match the language standard of the ROOT installation we are extending. The
  # C++ standard is also propagated by the imported ROOT targets, but the CUDA
  # standard has to be set explicitly. Both are only read when the target is
  # created further down, so setting them after project() is fine.
  if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD ${ROOT_CXX_STANDARD})
  endif()
  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
  endif()

  # libRooBatchCompute_CUDA is loaded at runtime by the RooBatchCompute
  # dispatcher with gSystem->Load("libRooBatchCompute_CUDA"), so it has to end
  # up in a directory that is part of ROOT's dynamic library search path. By
  # default we therefore install next to the ROOT installation we were
  # configured against. Relative paths are interpreted with respect to
  # CMAKE_INSTALL_PREFIX, which is what package managers want to use.
  # The cache entries are of type STRING and not PATH on purpose: CMake resolves
  # relative values of PATH cache entries given on the command line against the
  # current working directory, which is not what we want here.
  set(RooBatchCompute_CUDA_INSTALL_LIBDIR "${ROOT_LIBRARY_DIR}" CACHE STRING
      "Directory to install libRooBatchCompute_CUDA into. It must be in ROOT's dynamic library search path.")
  set(RooBatchCompute_CUDA_INSTALL_BINDIR "${ROOT_BINDIR}" CACHE STRING
      "Directory to install RooBatchCompute_CUDA.dll into (Windows only).")

  # Make sure the library finds libRooBatchCompute and libCore next to itself,
  # and also in the ROOT installation it was linked against.
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  if(APPLE)
    list(APPEND CMAKE_INSTALL_RPATH "@loader_path")
  else()
    list(APPEND CMAKE_INSTALL_RPATH "$ORIGIN")
  endif()

  message(STATUS "Building libRooBatchCompute_CUDA standalone against ROOT ${ROOT_VERSION} in ${ROOT_LIBRARY_DIR}")

endif()

if(NOT RooBatchCompute_STANDALONE_CUDA)

ROOT_LINKER_LIBRARY(RooBatchCompute
    src/Initialisation.cxx
  DEPENDENCIES
    Core
    MathCore
)

target_include_directories(RooBatchCompute PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/res>)

############################################################################################################################################
# Instantiations of the shared objects which provide the actual computation functions.

set(shared_object_sources src/RooBatchCompute.cxx src/ComputeFunctions.cxx)

# Generic implementation for CPUs that don't support vector instruction sets.
ROOT_LINKER_LIBRARY(RooBatchCompute_GENERIC ${shared_object_sources} TYPE SHARED DEPENDENCIES RooBatchCompute)
target_compile_options(RooBatchCompute_GENERIC  PRIVATE ${common-flags} -DRF_ARCH=GENERIC)

if(NOT CMAKE_VERSION VERSION_LESS "3.23.0")
  target_sources(
      RooBatchCompute
      PRIVATE
      FILE_SET private_header_files
      TYPE HEADERS
      BASE_DIRS res/
      FILES
          res/RooBatchCompute.h
          res/RooExprProgram.h
  )
  target_sources(
        RooBatchCompute_GENERIC
        PRIVATE
        FILE_SET private_header_files
        TYPE HEADERS
        BASE_DIRS src/ res/
        FILES
            res/RooBatchComputeTypes.h
            res/RooHeterogeneousMath.h
            res/RooNaNPacker.h
            src/Batches.h
            src/RooVDTHeaders.h
  )
endif()

# Windows platform and ICC compiler need special code and testing, thus the feature has not been implemented yet for these.
if (ROOT_PLATFORM MATCHES "linux|macosx" AND CMAKE_SYSTEM_PROCESSOR MATCHES x86_64 AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")

  target_compile_options(RooBatchCompute PRIVATE -DR__RF_ARCHITECTURE_SPECIFIC_LIBS)

  ROOT_LINKER_LIBRARY(RooBatchCompute_SSE4.1  ${shared_object_sources} TYPE SHARED DEPENDENCIES RooBatchCompute)
  ROOT_LINKER_LIBRARY(RooBatchCompute_AVX     ${shared_object_sources} TYPE SHARED DEPENDENCIES RooBatchCompute)
  ROOT_LINKER_LIBRARY(RooBatchCompute_AVX2    ${shared_object_sources} TYPE SHARED DEPENDENCIES RooBatchCompute)
  ROOT_LINKER_LIBRARY(RooBatchCompute_AVX512  ${shared_object_sources} TYPE SHARED DEPENDENCIES RooBatchCompute)

  # Flags -fno-signaling-nans, -fno-trapping-math and -O3 are necessary to enable autovectorization (especially for GCC).
  set(common-flags $<$<CXX_COMPILER_ID:GNU>:-fno-signaling-nans>)
  list(APPEND common-flags $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>: -fno-trapping-math -O3>)

  target_compile_options(RooBatchCompute_SSE4.1  PRIVATE ${common-flags} -msse4    -DRF_ARCH=SSE4)
  target_compile_options(RooBatchCompute_AVX     PRIVATE ${common-flags} -mavx     -DRF_ARCH=AVX)
  target_compile_options(RooBatchCompute_AVX2    PRIVATE ${common-flags} -mavx2    -DRF_ARCH=AVX2)
  target_compile_options(RooBatchCompute_AVX512  PRIVATE ${common-flags} -march=skylake-avx512 -DRF_ARCH=AVX512)

endif() # vector versions of library

endif() # NOT RooBatchCompute_STANDALONE_CUDA

if (cuda)
  set(shared_object_sources_cu src/RooBatchCompute.cu src/ComputeFunctions.cu src/CudaInterface.cu)

  if(RooBatchCompute_STANDALONE_CUDA)
    # No ROOT build tree around us, so ROOT_LINKER_LIBRARY() and the variables
    # it relies on are not available. Create the library by hand, with the same
    # name and layout, and link it against the installed libRooBatchCompute
    # (which owns the RooBatchCompute::dispatchCUDA pointer that the static
    # initializer of this library overwrites when it is loaded).
    add_library(RooBatchCompute_CUDA SHARED ${shared_object_sources_cu})
    # ROOT prefixes its libraries with "lib" on every platform, Windows
    # included, and that is the name gSystem->Load() is called with.
    set_target_properties(RooBatchCompute_CUDA PROPERTIES PREFIX lib IMPORT_PREFIX lib)
    target_include_directories(RooBatchCompute_CUDA
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/res ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(RooBatchCompute_CUDA PUBLIC ROOT::RooBatchCompute)
    install(TARGETS RooBatchCompute_CUDA
            LIBRARY DESTINATION ${RooBatchCompute_CUDA_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${RooBatchCompute_CUDA_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${RooBatchCompute_CUDA_INSTALL_BINDIR})
  else()
    ROOT_LINKER_LIBRARY(RooBatchCompute_CUDA  ${shared_object_sources_cu} TYPE SHARED DEPENDENCIES RooBatchCompute)
  endif()

  target_compile_options(RooBatchCompute_CUDA  PRIVATE -lineinfo --expt-relaxed-constexpr)
  if(NOT CMAKE_VERSION VERSION_LESS "3.23.0")
    target_sources(
        RooBatchCompute_CUDA
        PRIVATE
        FILE_SET private_header_files
        TYPE HEADERS
        BASE_DIRS src/
        FILES
            src/CudaInterface.h
    )
  endif()
endif()

if(vdt OR builtin_vdt)
  foreach(library RooBatchCompute_GENERIC RooBatchCompute_SSE4.1 RooBatchCompute_AVX RooBatchCompute_AVX2 RooBatchCompute_AVX512 )
    if(TARGET ${library})
      target_link_libraries(${library} PRIVATE VDT::VDT)
    endif()
  endforeach()
endif()
