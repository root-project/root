# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT graf2d/padpainter package
# @author Sergey Linev, GSI
############################################################################

set(PADPAINTER_HEADERS
  TPadPainter.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTPadPainter
  NO_INSTALL_HEADERS
  HEADERS
    ${PADPAINTER_HEADERS}
  SOURCES
    TPadPainter.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Gpad
    ASImageUtils
)

set(RELATIVE_INC_HEADERS ${PADPAINTER_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(ROOTPadPainter
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)

ROOT_INSTALL_HEADERS()
