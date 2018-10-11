############################################################################
# CMakeLists.txt file for building ROOT graf3d/g3d package
############################################################################

ROOT_GLOB_HEADERS(headers1 ${CMAKE_CURRENT_SOURCE_DIR}/inc/*.h)
ROOT_STANDARD_LIBRARY_PACKAGE(Graf3d
                              HEADERS ${headers1}
                              SOURCES *.cxx *.c
                              DEPENDENCIES Graf Hist Gpad MathCore)
