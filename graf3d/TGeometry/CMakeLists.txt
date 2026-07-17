# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT's TGeometry package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(TGeometry
  HEADERS
    TGeometry.h
    TMaterial.h
    TNode.h
    TRotMatrix.h
    TShape.h
    TMixture.h
    TBRIK.h
    TCONE.h
    TCONS.h
    TCTUB.h
    TELTU.h
    TGTRA.h
    THelix.h
    THYPE.h
    TNodeDiv.h
    TPARA.h
    TPCON.h
    TPGON.h
    TSPHE.h
    TTRAP.h
    TTRD1.h
    TTRD2.h
    TTUBE.h
    TTUBS.h
    TXTRU.h
  SOURCES
    TGeometry.cxx
    TMaterial.cxx
    TNode.cxx
    TRotMatrix.cxx
    TShape.cxx
    TMixture.cxx
    TBRIK.cxx
    TCONE.cxx
    TCONS.cxx
    TCTUB.cxx
    TELTU.cxx
    TGTRA.cxx
    THelix.cxx
    THYPE.cxx
    TNodeDiv.cxx
    TPARA.cxx
    TPCON.cxx
    TPGON.cxx
    TSPHE.cxx
    TTRAP.cxx
    TTRD1.cxx
    TTRD2.cxx
    TTUBE.cxx
    TTUBS.cxx
    TXTRU.cxx
  DEPENDENCIES
    Graf3d
)