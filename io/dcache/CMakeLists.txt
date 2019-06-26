# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT io/dcache package
############################################################################

include_directories(SYSTEM ${DCAP_INCLUDE_DIRS})

ROOT_STANDARD_LIBRARY_PACKAGE(DCache
                              HEADERS TDCacheFile.h
                              SOURCES src/TDCacheFile.cxx
                              LIBRARIES ${DCAP_LIBRARIES}
                              DEPENDENCIES Core Net RIO)
