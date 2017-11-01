
#
# FindZSTD.cmake
#
#
# The MIT License
#
# Copyright (c) 2016 MIT and Intel Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Finds the Zstandard library. This module defines:
#   - ZSTD_INCLUDE_DIR, directory containing headers
#   - ZSTD_LIBRARIES, the Zstandard library path
#   - ZSTD_FOUND, whether Zstandard has been found

# Find header files
if(ZSTD_SEARCH_HEADER_PATHS)
  find_path(
      ZSTD_INCLUDE_DIR zstd.h
      PATHS ${ZSTD_SEARCH_HEADER_PATHS}
      NO_DEFAULT_PATH
  )
else()
  find_path(ZSTD_INCLUDE_DIR zstd.h)
endif()

# Find library
if(ZSTD_SEARCH_LIB_PATH)
  find_library(
      ZSTD_LIBRARIES NAMES zstd
      PATHS ${ZSTD_SEARCH_LIB_PATH}$
      NO_DEFAULT_PATH
  )
else()
  find_library(ZSTD_LIBRARIES NAMES zstd)
endif()

if(ZSTD_INCLUDE_DIR AND ZSTD_LIBRARIES)
  message(STATUS "Found Zstandard: ${ZSTD_LIBRARIES}")
  set(ZSTD_FOUND TRUE)
else()
  set(ZSTD_FOUND FALSE)
endif()

if(ZSTD_FIND_REQUIRED AND NOT ZSTD_FOUND)
  message(FATAL_ERROR "Could not find the Zstandard library.")
endif()

if(ZSTD_FOUND)
  set(ZSTD_INCLUDE_DIRS "${ZSTD_INCLUDE_DIR}")

  if(NOT ZSTD_LIBRARIES)
    set(ZSTD_LIBRARIES ${ZSTD_LIBRARY})
  endif()

  if(NOT TARGET ZSTD::ZSTD)
    add_library(ZSTD::ZSTD UNKNOWN IMPORTED)
    set_target_properties(ZSTD::ZSTD PROPERTIES
      IMPORTED_LOCATION "${ZSTD_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${ZSTD_INCLUDE_DIRS}")
  endif()
endif()