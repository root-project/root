############################################################################
# CMakeLists.txt file for building ROOT core/lzma package
############################################################################

ROOT_OBJECT_LIBRARY(Lzma src/ZipLZMA.cxx BUILTINS LZMA)
target_include_directories(Lzma PRIVATE ${LZMA_INCLUDE_DIR} ${xxHash_INCLUDE_DIR})


ROOT_INSTALL_HEADERS()
