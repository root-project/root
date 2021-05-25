# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT gui/webdisplay package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTWebDisplay
    HEADERS 
       ROOT/RWebDisplayArgs.hxx
       ROOT/RWebDisplayHandle.hxx
       ROOT/RWebWindow.hxx
       ROOT/RWebWindowsManager.hxx
    SOURCES
       RWebDisplayArgs.cxx
       RWebDisplayHandle.cxx
       RWebWindow.cxx
       RWebWindowsManager.cxx
    DEPENDENCIES
       Core 
       RHTTP 
       MathCore
)
