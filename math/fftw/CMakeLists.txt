# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT math/fftw package
# @author Pere Mato, CERN
############################################################################

set(fftw_sources
    src/TFFTComplex.cxx
    src/TFFTComplexReal.cxx
    src/TFFTReal.cxx
    src/TFFTRealComplex.cxx
  )
set(fftw_headers
    TFFTComplex.h
    TFFTComplexReal.h
    TFFTReal.h
    TFFTRealComplex.h
  )

set(fftw_dependencies
  Core
  MathCore
  )

# Do we build this library standalone or as a component of ROOT?
string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" is_top_level)

# The ROOT binary distribution doesn't come with fftw3 enabled because of
# license incompatibilities (GPU). However, we want to make extending ROOT with
# the fftw3 plugin as easy as possible. Therefore, this CMake file supports
# standalone compilation of math/fftw.
if(is_top_level)

  cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

  # The standalone library will be called TFFT and not FFTW, to not confuse it
  # with the actual fftw3.
  set(fftw_libname TFFT)

  project(${fftw_libname} LANGUAGES CXX)

  find_package(ROOT REQUIRED COMPONENTS ${fftw_dependencies})
  include(${ROOT_USE_FILE}) # inherit ROOT's compile options, including C++ standard

  add_library(${fftw_libname} SHARED)
  target_sources(${fftw_libname} PRIVATE ${fftw_sources})
  target_include_directories(${fftw_libname} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/inc")
  target_link_libraries(${fftw_libname} PRIVATE fftw3)

  # Install the libraries and header files.
  install(TARGETS ${fftw_libname})

  set(CMAKE_INSTALL_INCLUDEDIR "include")
  ROOT_INSTALL_HEADERS()

else()

  ROOT_STANDARD_LIBRARY_PACKAGE(FFTW
    HEADERS ${fftw_headers}
    SOURCES ${fftw_sources}
    DEPENDENCIES ${fftw_dependencies}
    BUILTINS FFTW3
  )

  target_include_directories(FFTW PRIVATE ${FFTW_INCLUDE_DIR})
  target_link_libraries(FFTW PRIVATE ${FFTW_LIBRARIES})

endif()
