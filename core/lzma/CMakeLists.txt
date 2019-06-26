# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT core/lzma package
############################################################################

include_directories(${LZMA_INCLUDE_DIR})

ROOT_OBJECT_LIBRARY(Lzma src/ZipLZMA.c BUILTINS LZMA)

ROOT_INSTALL_HEADERS()
