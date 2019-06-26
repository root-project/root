# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT net/monalisa package
# @author Pere Mato, CERN
############################################################################

include_directories(${MONALISA_INCLUDE_DIR})

ROOT_STANDARD_LIBRARY_PACKAGE(Monalisa
  HEADERS
    TMonaLisaWriter.h
  SOURCES
    src/TMonaLisaWriter.cxx
  LIBRARIES
    ${MONALISA_LIBRARIES}
  DEPENDENCIES
    Net
    RIO
)
