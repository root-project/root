# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT core/lzma package
############################################################################

target_sources(Core PRIVATE src/ZipLZMA.c)

target_link_libraries(Core PRIVATE LibLZMA::LibLZMA)

target_include_directories(Core PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>)

ROOT_INSTALL_HEADERS()
