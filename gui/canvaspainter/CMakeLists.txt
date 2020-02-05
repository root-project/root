# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT gui/canvaspainter package
############################################################################

ROOT_LINKER_LIBRARY(ROOTCanvasPainter
  src/RCanvasPainter.cxx
  DEPENDENCIES
    ROOTGpadv7
    RHTTP
    ROOTWebDisplay
)
