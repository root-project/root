# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# - Locate jemalloc library
#
# Defines:
#
# JEMALLOC_FOUND
# JEMALLOC_LIBRARIES
# JEMALLOC_LIBRARY_PATH
# JEMALLOC_INCLUDE_DIR

find_path(JEMALLOC_ROOT_DIR NAMES include/jemalloc/jemalloc.h)
find_library(JEMALLOC_LIBRARIES NAMES jemalloc HINTS ${JEMALLOC_ROOT_DIR}/lib)
find_path(JEMALLOC_INCLUDE_DIR NAMES jemalloc/jemalloc.h HINTS ${JEMALLOC_ROOT_DIR}/include)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(jemalloc DEFAULT_MSG JEMALLOC_LIBRARIES JEMALLOC_INCLUDE_DIR)

mark_as_advanced(JEMALLOC_FOUND JEMALLOC_LIBRARIES JEMALLOC_INCLUDE_DIR)
get_filename_component(JEMALLOC_LIBRARY_PATH ${JEMALLOC_LIBRARIES} DIRECTORY)
