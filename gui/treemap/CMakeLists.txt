# Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT treemap package
# @author Patryk Tymoteusz Pilichowski CERN
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTTreeMap
  HEADERS
    ROOT/RTreeMapBase.hxx
    ROOT/RTreeMapPainter.hxx
  SOURCES
    RTreeMapBase.cxx
    RTreeMapImporter.cxx
    RTreeMapPainter.cxx
  DEPENDENCIES
    ROOTNTuple
    ROOTNTupleUtil
    Gpad
    RIO
)
