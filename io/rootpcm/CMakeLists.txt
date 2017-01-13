############################################################################
# CMakeLists.txt file for building ROOT io/rootpcm package
############################################################################

ROOT_GLOB_FILES(sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx)

ROOT_OBJECT_LIBRARY(RootPcmObjs ${sources})
