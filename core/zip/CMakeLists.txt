find_package(ZLIB REQUIRED)

ROOT_GLOB_HEADERS(headers
  inc/Compression.h
  inc/RZip.h
)

ROOT_GLOB_SOURCES(sources
  src/Bits.c
  src/ZDeflate.c
  src/ZTrees.c
  src/ZInflate.c
  src/Compression.cxx
  src/RZip.cxx
)

ROOT_OBJECT_LIBRARY(Zip ${sources})
target_include_directories(Zip PRIVATE ${ZLIB_INCLUDE_DIR})

ROOT_INSTALL_HEADERS()
