############################################################################
# CMakeLists.txt file for building ROOT math/minuit package
############################################################################

ROOT_USE_PACKAGE(math/matrix)
ROOT_USE_PACKAGE(math/mathcore)
ROOT_USE_PACKAGE(hist/hist)

ROOT_STANDARD_LIBRARY_PACKAGE(Minuit DEPENDENCIES Graf Hist Matrix MathCore)
