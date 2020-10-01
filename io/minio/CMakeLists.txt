# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT io/minio package
############################################################################

ROOT_LINKER_LIBRARY(RMiniIO
  src/RRawFile.cxx
  DEPENDENCIES
    Foundation
)

ROOT_INSTALL_HEADERS()
