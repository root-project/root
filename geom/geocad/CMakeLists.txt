############################################################################
# CMakeLists.txt file for building ROOT geom/geocad package
# @author Pere Mato, CERN
############################################################################

include_directories(${OCC_INCLUDE_DIR})

ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-overloaded-virtual)
ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-ignored-qualifiers)


ROOT_STANDARD_LIBRARY_PACKAGE(GeoCad
                              LIBRARIES ${OCC_LIBRARIES}
                              DEPENDENCIES Geom)
