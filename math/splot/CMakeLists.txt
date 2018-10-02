############################################################################
# CMakeLists.txt file for building ROOT math/splot package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(SPlot
  HEADERS
    TSPlot.h
  SOURCES
    src/TSPlot.cxx
  DEPENDENCIES
    Matrix
    Hist
    Tree
    TreePlayer
    Graf3d
    Graf
    MathCore
)
