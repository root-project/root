############################################################################
# CMakeLists.txt file for building ROOT math/fumili package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Fumili
  HEADERS
    TFumili.h
    TFumiliMinimizer.h
  SOURCES
    src/TFumili.cxx
    src/TFumiliMinimizer.cxx
  DEPENDENCIES
    Graf
    Hist
    MathCore
)
