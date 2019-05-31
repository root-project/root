############################################################################
# CMakeLists.txt file for building ROOT math/rtools package
# @author Pere Mato, CERN
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Rtools
  HEADERS
    Math/RMinimizer.h
  SOURCES
    src/RMinimizer.cxx
  DICTIONARY_OPTIONS
    -I${R_INCLUDE_DIR}
  DEPENDENCIES
    Core
    MathCore
    RInterface
)

target_include_directories(Rtools BEFORE PRIVATE ${R_INCLUDE_DIR})

file(COPY "${CMAKE_SOURCE_DIR}/etc/plugins/ROOT@@Math@@Minimizer/P090_RMinimizer.C"
  DESTINATION "${CMAKE_BINARY_DIR}/etc/plugins/ROOT@@Math@@Minimizer/")
