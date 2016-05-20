############################################################################
# CMakeLists.txt file for building ROOT core/lzo package
############################################################################

#---Declare ZipLZO sources as part of libCore-------------------------------
set(sources ${CMAKE_CURRENT_SOURCE_DIR}/src/ZipLZO.c)


ROOT_OBJECT_LIBRARY(Lzo ${sources})

ROOT_INSTALL_HEADERS()
