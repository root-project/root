# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

find_package(ZLIB REQUIRED)


ROOT_OBJECT_LIBRARY(Zip
  src/Bits.c
  src/ZDeflate.c
  src/ZTrees.c
  src/ZInflate.c
  src/Compression.cxx
  src/RZip.cxx
  INCLUDES
    ${ZLIB_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/core/lzma/inc
    ${CMAKE_SOURCE_DIR}/core/lz4/inc
    ${CMAKE_SOURCE_DIR}/core/zstd/inc
)

ROOT_INSTALL_HEADERS()
