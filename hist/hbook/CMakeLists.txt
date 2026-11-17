# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT hist/hbook package
############################################################################

set(HBOOK_HEADERS
  THbookBranch.h
  THbookFile.h
  THbookKey.h
  THbookTree.h
)

set(RELATIVE_INC_HEADERS ${HBOOK_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)

ROOT_STANDARD_LIBRARY_PACKAGE(Hbook
  HEADERS
    ${HBOOK_HEADERS}
  SOURCES
    src/THbookBranch.cxx
    src/THbookFile.cxx
    src/THbookKey.cxx
    src/THbookTree.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  LIBRARIES
    minicern
  DEPENDENCIES
    Hist
    Tree
    Graf
    TreePlayer
    RIO
)

target_sources(Hbook
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)

# This is needed in particular for macOS, where the path of the GNU Fortran
# library might not be in the default DYLD_LIBRARY_PATH, but it gets picked
# up with enable_language(Fortran).
if(APPLE AND CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  # Compute path to quadmath
  execute_process(
    COMMAND ${CMAKE_Fortran_COMPILER} -print-file-name=libgfortran.dylib
    OUTPUT_VARIABLE LIBGFORTRAN_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  get_filename_component(LIBGFORTRAN_DIR "${LIBGFORTRAN_PATH}" DIRECTORY)
  set_property(TARGET Hbook APPEND PROPERTY BUILD_RPATH "${LIBGFORTRAN_DIR}")
  set_property(TARGET Hbook APPEND PROPERTY INSTALL_RPATH "${LIBGFORTRAN_DIR}")
endif()
