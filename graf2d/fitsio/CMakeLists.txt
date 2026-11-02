# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT graf2d/fitsio package
# @author Pere Mato, CERN
############################################################################

set(FITSIO_HEADERS TFITS.h)

ROOT_STANDARD_LIBRARY_PACKAGE(FITSIO
  HEADERS
    ${FITSIO_HEADERS}
  SOURCES
    TFITS.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  LIBRARIES
    CFITSIO::CFITSIO
  DEPENDENCIES
    Gpad
    Graf
    Hist
    Matrix
)

set(RELATIVE_INC_HEADERS ${FITSIO_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(FITSIO
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)
