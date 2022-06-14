# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT geom/gdml package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Gdml
  HEADERS
    TGDMLParse.h
    TGDMLWrite.h
  SOURCES
    src/TGDMLParse.cxx
    src/TGDMLWrite.cxx
  DEPENDENCIES
    Geom
    XMLIO
    Hist
    RIO
)
