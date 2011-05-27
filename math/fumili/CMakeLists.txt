############################################################################
# CMakeLists.txt file for building ROOT math/fumili package
############################################################################

ROOT_USE_PACKAGE(math/matrix)
ROOT_USE_PACKAGE(math/mathcore)
ROOT_USE_PACKAGE(hist/hist)

ROOT_STANDARD_LIBRARY_PACKAGE(Fumili DEPENDENCIES Graf Hist MathCore)


