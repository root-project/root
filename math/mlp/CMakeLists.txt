############################################################################
# CMakeLists.txt file for building ROOT math/physics package
############################################################################

ROOT_USE_PACKAGE(tree/tree)
ROOT_USE_PACKAGE(tree/treeplayer)
ROOT_USE_PACKAGE(hist/hist)

ROOT_STANDARD_LIBRARY_PACKAGE(MLP DEPENDENCIES Hist Matrix Tree Graf Gpad TreePlayer MathCore)

