find_package(ZLIB REQUIRED)
find_package(ZSTD REQUIRED)

ROOT_OBJECT_LIBRARY(Zip
  src/Bits.c
  src/ZDeflate.c
  src/ZTrees.c
  src/ZInflate.c
  src/Compression.cxx
  src/RZip.cxx
)

target_include_directories(Zip PRIVATE ${ZLIB_INCLUDE_DIR} ${ZSTD_INCLUDE_DIR})

ROOT_INSTALL_HEADERS()
