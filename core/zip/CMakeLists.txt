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
)

target_include_directories(Zip PRIVATE ${ZLIB_INCLUDE_DIR})

ROOT_INSTALL_HEADERS()
