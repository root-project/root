############################################################################
# CMakeLists.txt file for building ROOT montecarlo/vmc package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(montecarlo/eg)
ROOT_USE_PACKAGE(geom/geom)

ROOT_STANDARD_LIBRARY_PACKAGE(VMC DEPENDENCIES EG Geom MathCore)
