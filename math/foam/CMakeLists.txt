############################################################################
# CMakeLists.txt file for building ROOT math/foam package
############################################################################

ROOT_USE_PACKAGE(hist/hist)

ROOT_STANDARD_LIBRARY_PACKAGE(Foam DEPENDENCIES Hist MathCore)

