# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT graf2d/quartz package
# @author Pere Mato, CERN
############################################################################

add_definitions("-ObjC++")

ROOT_STANDARD_LIBRARY_PACKAGE(GQuartz
  NO_HEADERS
  SOURCES
    src/QuartzFillArea.mm
    src/QuartzLine.mm
    src/QuartzMarker.mm
    src/QuartzText.mm
    src/QuartzUtils.mm
  LIBRARIES
    "-framework Cocoa"
  DEPENDENCIES
    Core
)
