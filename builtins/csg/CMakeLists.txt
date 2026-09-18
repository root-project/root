# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

add_library(ROOTCsg STATIC)
target_sources(ROOTCsg PRIVATE src/CsgOps.cxx)
target_include_directories(ROOTCsg PUBLIC inc/)
set_property(TARGET ROOTCsg PROPERTY POSITION_INDEPENDENT_CODE ON)
