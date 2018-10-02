############################################################################
# CMakeLists.txt file for building ROOT math/mlp package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(MLP
  HEADERS
    TMLPAnalyzer.h
    TMultiLayerPerceptron.h
    TNeuron.h
    TSynapse.h
  SOURCES
    src/TMLPAnalyzer.cxx
    src/TMultiLayerPerceptron.cxx
    src/TNeuron.cxx
    src/TSynapse.cxx
  DEPENDENCIES
    Hist
    Matrix
    Tree
    Graf
    Gpad
    TreePlayer
    MathCore
)
