# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

find_package(ZLIB REQUIRED)

target_sources(Core PRIVATE
  src/Bits.c
  src/ZDeflate.c
  src/ZTrees.c
  src/ZInflate.c
  src/Compression.cxx
  src/RZip.cxx
)

target_link_libraries(Core PRIVATE ZLIB::ZLIB)

target_include_directories(Core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

ROOT_INSTALL_HEADERS()
