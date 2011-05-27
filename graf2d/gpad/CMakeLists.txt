############################################################################
# CMakeLists.txt file for building ROOT graf2d/gpad package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(graf2d/graf)
ROOT_USE_PACKAGE(core)
include_directories(${CMAKE_SOURCE_DIR}/hist/hist/inc)  # This is to avoid a circular dependency graf <--> hist 

ROOT_STANDARD_LIBRARY_PACKAGE(Gpad DEPENDENCIES Graf Hist)

