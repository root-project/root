############################################################################
# CMakeLists.txt file for building ROOT graf3d/g3d package
############################################################################

ROOT_GLOB_HEADERS(headers1 ${CMAKE_CURRENT_SOURCE_DIR}/inc/*.h)
list(REMOVE_ITEM headers1 ${CMAKE_CURRENT_SOURCE_DIR}/inc/X3DBuffer.h)
set(headers2 X3DBuffer.h )

ROOT_STANDARD_LIBRARY_PACKAGE(Graf3d
                              HEADERS ${headers1}
                              SOURCES *.cxx *.c
                              DEPENDENCIES Graf Hist Gpad MathCore)
