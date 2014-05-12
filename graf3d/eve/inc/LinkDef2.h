// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006 - 2009

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//==============================================================================
// LinkDef2.h - Visualization elements and helper classes.
//==============================================================================

// TEveTrack
#pragma link C++ class TEveTrack+;
#pragma link C++ class TEveTrackGL+;
#pragma link C++ class TEveTrackEditor+;
#pragma link C++ class TEveTrackList+;
#pragma link C++ class TEveTrackListEditor+;
#pragma link C++ class TEveTrackProjected+;
#pragma link C++ class TEveTrackProjectedGL+;
#pragma link C++ class TEveTrackListProjected+;

// TEveTrackPropagator
#pragma link C++ class TEveTrackPropagator+;
#pragma link C++ class TEveTrackPropagatorSubEditor+;
#pragma link C++ class TEveTrackPropagatorEditor+;
#pragma link C++ class TEveMagField+;
#pragma link C++ class TEveMagFieldConst+;
#pragma link C++ class TEveMagFieldDuo+;

// TEveText
#pragma link C++ class TEveText+;
#pragma link C++ class TEveTextGL+;
#pragma link C++ class TEveTextEditor+;

// TEvePointSet
#pragma link C++ class TEvePointSet+;
#pragma link C++ class TEvePointSetArray+;
#pragma link C++ class TEvePointSetArrayEditor+;
#pragma link C++ class TEvePointSetProjected+;

// TEveLine
#pragma link C++ class TEveLine+;
#pragma link C++ class TEveLineEditor+;
#pragma link C++ class TEveLineGL+;
#pragma link C++ class TEveLineProjected+;

// TEveArrow
#pragma link C++ class TEveArrow+;
#pragma link C++ class TEveArrowEditor+;
#pragma link C++ class TEveArrowGL+;

// TEveDigitSet
#pragma link C++ class TEveDigitSet+;
#pragma link C++ class TEveDigitSetEditor+;
#pragma link C++ class TEveDigitSetGL+;
// #pragma link C++ typedef TEveDigitSet::Callback_foo;
// #pragma link C++ typedef TEveDigitSet::TooltipCB_foo;

// TEveQuadSet
#pragma link C++ class TEveQuadSet+;
#pragma link C++ class TEveQuadSetGL+;

// TEveBoxSet
#pragma link C++ class TEveBoxSet+;
#pragma link C++ class TEveBoxSetGL+;

// TEveGeoNode
#pragma link C++ class TEveGeoNode+;
#pragma link C++ class TEveGeoTopNode+;
#pragma link C++ class TEveGeoNodeEditor+;
#pragma link C++ class TEveGeoTopNodeEditor+;
#pragma link C++ class TEveGeoShape+;
#pragma link C++ class TEveGeoShapeProjected+;

// TEveGeoShapeExtract
#pragma link C++ class TEveGeoShapeExtract+;

// Arbitrary-tesselation TGeoShape.
#pragma link C++ class TEveGeoPolyShape+;

// Various shapes
#pragma link C++ class TEveShape+;
#pragma link C++ class TEveShapeEditor+;
#pragma link C++ class TEveBox+;
#pragma link C++ class TEveBoxGL+;
#pragma link C++ class TEveBoxProjected+;
#pragma link C++ class TEveBoxProjectedGL+;

// TEvePolygonSetProjected
#pragma link C++ class TEvePolygonSetProjected+;
#pragma link C++ class TEvePolygonSetProjectedGL+;

// TEveTrianlgeSet
#pragma link C++ class TEveTriangleSet+;
#pragma link C++ class TEveTriangleSetEditor+;
#pragma link C++ class TEveTriangleSetGL+;

// TEveStraightLineSet
#pragma link C++ class TEveStraightLineSet+;
#pragma link C++ class TEveStraightLineSetGL+;
#pragma link C++ class TEveStraightLineSetEditor+;
#pragma link C++ class TEveStraightLineSetProjected+;
#pragma link C++ class TEveScalableStraightLineSet+;

// TEveCalo
#pragma link C++ class TEveCaloData+;
#pragma link C++ class TEveCaloData::SliceInfo_t+;
#pragma link C++ class TEveCaloDataVec;
#pragma link C++ class TEveCaloDataHist+;
#pragma link C++ class TEveCaloViz+;
#pragma link C++ class TEveCaloVizEditor+;
#pragma link C++ class TEveCalo3D+;
#pragma link C++ class TEveCalo3DEditor+;
#pragma link C++ class TEveCalo3DGL+;
#pragma link C++ class TEveCalo2D+;
#pragma link C++ class TEveCalo2DGL+;
#pragma link C++ class TEveCaloLego+;
#pragma link C++ class TEveCaloLegoEditor+;
#pragma link C++ class TEveCaloLegoGL+;
#pragma link C++ class TEveCaloLegoOverlay+;

// TEveLegoEventHandler
#pragma link C++ class TEveLegoEventHandler+;

// TEveJetCone
#pragma link C++ class TEveJetCone+;
#pragma link C++ class TEveJetConeEditor+;
#pragma link C++ class TEveJetConeGL+;
#pragma link C++ class TEveJetConeProjected+;
#pragma link C++ class TEveJetConeProjectedGL+;

// TEvePlots
#pragma link C++ class TEvePlot3D+;
#pragma link C++ class TEvePlot3DGL+;

// TEveFrameBox
#pragma link C++ class TEveFrameBox+;
#pragma link C++ class TEveFrameBoxGL+;

// TEveGridStepper
#pragma link C++ class TEveGridStepper+;
#pragma link C++ class TEveGridStepperSubEditor+;
#pragma link C++ class TEveGridStepperEditor+;

// TEveRGBAPalette
#pragma link C++ class TEveRGBAPalette+;
#pragma link C++ class TEveRGBAPaletteEditor+;
#pragma link C++ class TEveRGBAPaletteSubEditor+;
#pragma link C++ class TEveRGBAPaletteOverlay+;
