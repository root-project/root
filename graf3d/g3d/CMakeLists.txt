# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT graf3d/g3d package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Graf3d
  HEADERS
    TAxis3D.h
    TGeometryTransformer.h
    TMarker3DBox.h
    TPoints3DABC.h
    TPointSet3D.h
    TPolyLine3D.h
    TPolyMarker3D.h
    TView3D.h
    X3DBuffer.h
  SOURCES
    TAxis3D.cxx
    TMarker3DBox.cxx
    TPoints3DABC.cxx
    TPointSet3D.cxx
    TPolyLine3D.cxx
    TPolyMarker3D.cxx
    TView3D.cxx
    X3DBuffer.c
  DEPENDENCIES
    Gpad
    Graf
    Hist
    MathCore
)
