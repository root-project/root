############################################################################
# CMakeLists.txt file for building ROOT math/splot package
############################################################################

ROOT_USE_PACKAGE(hist/hist)
ROOT_USE_PACKAGE(tree/tree)
ROOT_USE_PACKAGE(tree/treeplayer)
ROOT_USE_PACKAGE(graf3d)

ROOT_STANDARD_LIBRARY_PACKAGE(SPlot DEPENDENCIES Matrix Hist Tree TreePlayer Graf3d Graf MathCore)

