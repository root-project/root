# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT io/rootpcm package
############################################################################

ROOT_OBJECT_LIBRARY(RootPcmObjs src/rootclingIO.cxx)

target_include_directories(RootPcmObjs PRIVATE
   ${PROJECT_SOURCE_DIR}/io/io/inc
   ${PROJECT_SOURCE_DIR}/core/zip/inc
)
