// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#pragma link off all functions;
#pragma link off all globals;
#pragma link off all classes;


// Utilities
#pragma link C++ class TEveUtil+;

// TEveManager
#pragma link C++ class TEveManager+;
#pragma link C++ global gEve;

#pragma link C++ class TEveManager::TExceptionHandler+;

// Basic helper classes
#pragma link C++ class TEveException+;
#pragma link C++ class TEvePadHolder+;
#pragma link C++ class TEveGeoManagerHolder+;
#pragma link C++ class TEveRefCnt+;
#pragma link C++ class TEveRefBackPtr+;

// TEveVSD structs
#pragma link C++ class TEveVector+;
#pragma link C++ class TEvePathMark+;
#pragma link C++ class TEveMCTrack+;
#pragma link C++ class TEveHit+;
#pragma link C++ class TEveCluster+;
#pragma link C++ class TEveRecTrack+;
#pragma link C++ class TEveRecKink+;
#pragma link C++ class TEveRecV0+;
#pragma link C++ class TEveMCRecCrossRef+;

// TEveTrans
#pragma link C++ class TEveTrans-;
#pragma link C++ class TEveTransSubEditor+;
#pragma link C++ class TEveTransEditor+;

// Stepper
#pragma link C++ class TEveGridStepper+;
#pragma link C++ class TEveGridStepperSubEditor+;
#pragma link C++ class TEveGridStepperEditor+;

// TEveRGBAPalette
#pragma link C++ class TEveRGBAPalette+;
#pragma link C++ class TEveRGBAPaletteEditor+;
#pragma link C++ class TEveRGBAPaletteSubEditor+;

// Plexes
#pragma link C++ class TEveChunkManager+;
#pragma link C++ class TEveChunkManager::iterator-;

// TEveEventManager, VSDEvent, TEveVSD
#pragma link C++ class TEveEventManager+;
#pragma link C++ class TEveVSD+;

// TTreeTools
#pragma link C++ class TEveSelectorToEventList+;
#pragma link C++ class TEvePointSelectorConsumer+;
#pragma link C++ class TEvePointSelector+;

// TEveElement
#pragma link C++ class TEveElement+;
#pragma link C++ class TEveElement::TEveListTreeInfo+;
#pragma link C++ class TEveElementObjectPtr+;
#pragma link C++ class TEveElementList+;
#pragma link C++ class TEveElementEditor+;

#pragma link C++ class std::list<TEveElement*>;
#pragma link C++ class std::list<TEveElement*>::iterator;
#pragma link C++ typedef TEveElement::List_t;
#pragma link C++ typedef TEveElement::List_i;

#pragma link C++ class std::set<TEveElement*>;
#pragma link C++ class std::set<TEveElement*>::iterator;
#pragma link C++ typedef TEveElement::Set_t;
#pragma link C++ typedef TEveElement::Set_i;

// TEveSelection
#pragma link C++ class TEveSelection+;

// GL-interface
#pragma link C++ class TEveScene+;
#pragma link C++ class TEveSceneList+;
#pragma link C++ class TEveSceneInfo+;
#pragma link C++ class TEveViewer+;
#pragma link C++ class TEveViewerList+;

// TEvePad
#pragma link C++ class TEvePad+;

// TEveBrowser
#pragma link C++ class TEveListTreeItem+;
#pragma link C++ class TEveGListTreeEditorFrame+;
#pragma link C++ class TEveBrowser+;

// TEveGedEditor
#pragma link C++ class TEveGedEditor+;

// TEveMacro
#pragma link C++ class TEveMacro+;

// RGValuators
#pragma link C++ class TEveGValuatorBase+;
#pragma link C++ class TEveGValuator+;
#pragma link C++ class TEveGDoubleValuator+;
#pragma link C++ class TEveGTriVecValuator+;

// TEveTrack
#pragma link C++ class TEveTrack+;
#pragma link C++ class TEveTrackGL+;
#pragma link C++ class TEveTrackEditor+;
#pragma link C++ class TEveTrackList+;
#pragma link C++ class TEveTrackListEditor+;
#pragma link C++ class TEveTrackPropagatorSubEditor+;
#pragma link C++ class TEveTrackPropagatorEditor+;
#pragma link C++ class TEveTrackPropagator+;
#pragma link C++ class TEveTrackCounter+;
#pragma link C++ class TEveTrackCounterEditor+;


// TEveText
#pragma link C++ class TEveText+;
#pragma link C++ class TEveTextGL+;
#pragma link C++ class TEveTextEditor+;

// TEvePointSet
#pragma link C++ class TEvePointSet+;
#pragma link C++ class TEvePointSetArray+;
#pragma link C++ class TEvePointSetArrayEditor+;

// TEveLine
#pragma link C++ class TEveLine+;
#pragma link C++ class TEveLineEditor+;
#pragma link C++ class TEveLineGL+;

// TEveFrameBox
#pragma link C++ class TEveFrameBox+;
#pragma link C++ class TEveFrameBoxGL+;

// TEveDigitSet
#pragma link C++ class TEveDigitSet+;
#pragma link C++ class TEveDigitSetEditor+;

// TEveQuadSet
#pragma link C++ class TEveQuadSet+;
#pragma link C++ class TEveQuadSetGL+;

// TEveBoxSet
#pragma link C++ class TEveBoxSet+;
#pragma link C++ class TEveBoxSetGL+;

// GeoNode
#pragma link C++ class TEveGeoNode+;
#pragma link C++ class TEveGeoTopNode+;
#pragma link C++ class TEveGeoNodeEditor+;
#pragma link C++ class TEveGeoTopNodeEditor+;

#pragma link C++ class TEveGeoShapeExtract+;
#pragma link C++ class TEveGeoShape+;

// TrianlgeSet
#pragma link C++ class TEveTriangleSet+;
#pragma link C++ class TEveTriangleSetEditor+;
#pragma link C++ class TEveTriangleSetGL+;

// TEveStraightLineSet
#pragma link C++ class TEveStraightLineSet+;
#pragma link C++ class TEveStraightLineSetGL+;
#pragma link C++ class TEveStraightLineSetEditor+;
#pragma link C++ class TEveStraightLineSetProjected+;

// Projections / non-linear transformations
#pragma link C++ class TEveProjectable+;
#pragma link C++ class TEveProjected+;
#pragma link C++ class TEveProjection+;
#pragma link C++ class TEveRhoZProjection+;
#pragma link C++ class TEveRPhiProjection+;

#pragma link C++ class TEveProjectionManager+;
#pragma link C++ class TEveProjectionManagerEditor+;
#pragma link C++ class TEveProjectionAxes+;
#pragma link C++ class TEveProjectionAxesEditor+;
#pragma link C++ class TEveProjectionAxesGL+;

#pragma link C++ class TEvePointSetProjected+;
#pragma link C++ class TEveLineProjected+;
#pragma link C++ class TEveTrackProjected+;
#pragma link C++ class TEveTrackProjectedGL+;
#pragma link C++ class TEveTrackListProjected+;

#pragma link C++ class TEvePolygonSetProjected+;
#pragma link C++ class TEvePolygonSetProjectedEditor+;
#pragma link C++ class TEvePolygonSetProjectedGL+;

// Generic calorimeter representation
#pragma link C++ class TEveCaloData+;
#pragma link C++ class TEveCaloDataHist+;
#pragma link C++ class TEveCaloViz+;
#pragma link C++ class TEveCaloVizEditor+;
#pragma link C++ class TEveCalo3D+;
#pragma link C++ class TEveCalo3DGL+;
#pragma link C++ class TEveCalo2D+;
#pragma link C++ class TEveCalo2DGL+;
#pragma link C++ class TEveCaloLego+;
#pragma link C++ class TEveCaloLegoEditor+;
#pragma link C++ class TEveCaloLegoGL+;

// Generic configuration
#pragma link C++ class TEveParamList;
#pragma link C++ class TEveParamList::FloatConfig_t+;
#pragma link C++ class TEveParamList::IntConfig_t+;
#pragma link C++ class TEveParamList::BoolConfig_t+;
#pragma link C++ class TEveParamListEditor+;
