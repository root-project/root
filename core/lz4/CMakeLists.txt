# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

find_package(LZ4 REQUIRED)
find_package(xxHash REQUIRED)

target_sources(Core PRIVATE src/ZipLZ4.cxx)
target_link_libraries(Core PRIVATE xxHash::xxHash LZ4::LZ4)
target_include_directories(Core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

ROOT_INSTALL_HEADERS()
