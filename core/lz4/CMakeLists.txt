find_package(LZ4 REQUIRED)
find_package(xxHash REQUIRED)

ROOT_GLOB_HEADERS(headers inc/ZipLZ4.h)
ROOT_GLOB_SOURCES(sources src/ZipLZ4.cxx)

ROOT_OBJECT_LIBRARY(Lz4 ${sources})
target_include_directories(Lz4 PRIVATE ${LZ4_INCLUDE_DIR} ${xxHash_INCLUDE_DIR})

ROOT_INSTALL_HEADERS()
