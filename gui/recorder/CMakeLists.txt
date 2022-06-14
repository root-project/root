# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT gui/recorder package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Recorder
  HEADERS
    TRecorder.h
  SOURCES
    src/TRecorder.cxx
  DEPENDENCIES
    RIO
    Tree
    Gui
    Thread
)
