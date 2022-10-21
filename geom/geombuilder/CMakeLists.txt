# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT geom/geom package
# @author Pere Mato, CERN
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(GeomBuilder
  HEADERS
    TGeoBBoxEditor.h
    TGeoConeEditor.h
    TGeoEltuEditor.h
    TGeoGedFrame.h
    TGeoHypeEditor.h
    TGeoManagerEditor.h
    TGeoMaterialEditor.h
    TGeoMatrixEditor.h
    TGeoMediumEditor.h
    TGeoNodeEditor.h
    TGeoParaEditor.h
    TGeoPconEditor.h
    TGeoPgonEditor.h
    TGeoSphereEditor.h
    TGeoTabManager.h
    TGeoTorusEditor.h
    TGeoTrapEditor.h
    TGeoTrd1Editor.h
    TGeoTrd2Editor.h
    TGeoTubeEditor.h
    TGeoVolumeEditor.h
  SOURCES
    src/TGeoBBoxEditor.cxx
    src/TGeoConeEditor.cxx
    src/TGeoEltuEditor.cxx
    src/TGeoGedFrame.cxx
    src/TGeoHypeEditor.cxx
    src/TGeoManagerEditor.cxx
    src/TGeoMaterialEditor.cxx
    src/TGeoMatrixEditor.cxx
    src/TGeoMediumEditor.cxx
    src/TGeoNodeEditor.cxx
    src/TGeoParaEditor.cxx
    src/TGeoPconEditor.cxx
    src/TGeoPgonEditor.cxx
    src/TGeoSphereEditor.cxx
    src/TGeoTabManager.cxx
    src/TGeoTorusEditor.cxx
    src/TGeoTrapEditor.cxx
    src/TGeoTrd1Editor.cxx
    src/TGeoTrd2Editor.cxx
    src/TGeoTubeEditor.cxx
    src/TGeoVolumeEditor.cxx
  DEPENDENCIES
    Geom
    Graf3d
    Gpad
    Graf
    ROOTGui
    Ged
)
