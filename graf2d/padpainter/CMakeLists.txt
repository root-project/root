# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT graf2d/padpainter package
# @author Sergey Linev, GSI
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTPadPainter
  NO_INSTALL_HEADERS
  HEADERS
    TPadPainter.h
  SOURCES
    TPadPainter.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Gpad
)


ROOT_INSTALL_HEADERS()
