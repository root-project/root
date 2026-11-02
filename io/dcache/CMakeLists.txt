# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT io/dcache package
############################################################################

set(DCACHE_HEADERS TDCacheFile.h)

ROOT_STANDARD_LIBRARY_PACKAGE(DCache
                              HEADERS ${DCACHE_HEADERS}
                              SOURCES src/TDCacheFile.cxx
                              LIBRARIES ${DCAP_LIBRARIES}
                              DEPENDENCIES Core Net RIO)

set(RELATIVE_INC_HEADERS ${DCACHE_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)

target_sources(DCache
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)

target_include_directories(DCache PRIVATE ${DCAP_INCLUDE_DIRS})
