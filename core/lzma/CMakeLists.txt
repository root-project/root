############################################################################
# CMakeLists.txt file for building ROOT core/lzma package
############################################################################

include_directories(${LZMA_INCLUDE_DIR})

ROOT_OBJECT_LIBRARY(Lzma src/ZipLZMA.c BUILTINS LZMA)

ROOT_INSTALL_HEADERS()
