############################################################################
# CMakeLists.txt file for building ROOT math/minuit package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Minuit
  HEADERS
    TFitter.h
    TLinearFitter.h
    TLinearMinimizer.h
    TMinuit.h
    TMinuitMinimizer.h
 SOURCES
    src/TFitter.cxx
    src/TLinearFitter.cxx
    src/TLinearMinimizer.cxx
    src/TMinuit.cxx
    src/TMinuitMinimizer.cxx
  DEPENDENCIES
    Graf
    Hist
    Matrix
    MathCore
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
)

if (CMAKE_BUILD_TYPE STREQUAL Optimized)
  target_compile_options(Minuit PRIVATE "-fno-fast-math")
endif()
