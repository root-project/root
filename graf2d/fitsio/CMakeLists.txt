# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT graf2d/fitsio package
# @author Pere Mato, CERN
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(FITSIO
  HEADERS
    TFITS.h
  SOURCES
    TFITS.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  LIBRARIES
    ${CFITSIO_LIBRARIES}
  DEPENDENCIES
    Gpad
    Graf
    Hist
    Matrix
  BUILTINS
    CFITSIO
)

target_include_directories(FITSIO PRIVATE ${CFITSIO_INCLUDE_DIR})
