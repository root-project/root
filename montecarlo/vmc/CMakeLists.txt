############################################################################
# CMakeLists.txt file for building ROOT montecarlo/vmc package
# @author Pere Mato, CERN
############################################################################

if(geo)
  ROOT_STANDARD_LIBRARY_PACKAGE(VMC DEPENDENCIES EG Geom MathCore)
endif()
