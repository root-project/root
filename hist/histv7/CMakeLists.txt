# Support standalone builds of this package.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  # These two statements must be in the top-level CMakeLists.txt...
  cmake_minimum_required(VERSION 3.20)
  project(ROOTHist)

  # The rest goes into a separate file, to keep the setup clean.
  include(standalone.cmake)
  return()
endif()

include(headers.cmake)

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTHist
  HEADERS
    ${histv7_headers}
  NO_SOURCES
  DEPENDENCIES
    Core
)

# Function used in test/ to add unitest executables.
function(HIST_ADD_GTEST name source)
  ROOT_ADD_GTEST(${name} ${source} LIBRARIES ROOTHist)
endfunction()

ROOT_ADD_TEST_SUBDIRECTORY(test)

if(DEFINED histv7_benchmark AND histv7_benchmark)
  # The user set the "hidden" flag, require the benchmark library.
  find_package(benchmark CONFIG REQUIRED)
  # Keep the binaries out of the global bin/ directory.
  unset(CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  add_subdirectory(benchmark)
endif()
