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
// LinkDef1.h - Core EVE objects and services.
//==============================================================================

// TEveVector
#pragma link C++ class   ROOT::Experimental::TEveVectorT<Float_t>+;
#pragma link C++ class   ROOT::Experimental::TEveVectorT<Double_t>+;
#pragma link C++ typedef ROOT::Experimental::TEveVector;
#pragma link C++ typedef ROOT::Experimental::TEveVectorF;
#pragma link C++ typedef ROOT::Experimental::TEveVectorD;

#pragma link C++ class   ROOT::Experimental::TEveVector4T<Float_t>+;
#pragma link C++ class   ROOT::Experimental::TEveVector4T<Double_t>+;
#pragma link C++ typedef ROOT::Experimental::TEveVector4;
#pragma link C++ typedef ROOT::Experimental::TEveVector4F;
#pragma link C++ typedef ROOT::Experimental::TEveVector4D;

#pragma link C++ class   ROOT::Experimental::TEveVector2T<Float_t>+;
#pragma link C++ class   ROOT::Experimental::TEveVector2T<Double_t>+;
#pragma link C++ typedef ROOT::Experimental::TEveVector2;
#pragma link C++ typedef ROOT::Experimental::TEveVector2F;
#pragma link C++ typedef ROOT::Experimental::TEveVector2D;

// Operators for TEveVectorXT<Float_t>
#pragma link C++ function operator+(const ROOT::Experimental::TEveVectorT<Float_t>&, const ROOT::Experimental::TEveVectorT<Float_t>&);
#pragma link C++ function operator-(const ROOT::Experimental::TEveVectorT<Float_t>&, const ROOT::Experimental::TEveVectorT<Float_t>&);
#pragma link C++ function operator*(const ROOT::Experimental::TEveVectorT<Float_t>&, Float_t);
#pragma link C++ function operator*(Float_t, const ROOT::Experimental::TEveVectorT<Float_t>&);
#pragma link C++ function operator+(const ROOT::Experimental::TEveVector4T<Float_t>&, const ROOT::Experimental::TEveVector4T<Float_t>&);
#pragma link C++ function operator-(const ROOT::Experimental::TEveVector4T<Float_t>&, const ROOT::Experimental::TEveVector4T<Float_t>&);
#pragma link C++ function operator*(const ROOT::Experimental::TEveVector4T<Float_t>&, Float_t);
#pragma link C++ function operator*(Float_t, const ROOT::Experimental::TEveVector4T<Float_t>&);
#pragma link C++ function operator+(const ROOT::Experimental::TEveVector2T<Float_t>&, const ROOT::Experimental::TEveVector2T<Float_t>&);
#pragma link C++ function operator-(const ROOT::Experimental::TEveVector2T<Float_t>&, const ROOT::Experimental::TEveVector2T<Float_t>&);
#pragma link C++ function operator*(const ROOT::Experimental::TEveVector2T<Float_t>&, Float_t);
#pragma link C++ function operator*(Float_t, const ROOT::Experimental::TEveVector2T<Float_t>&);
// Operators for TEveVectorXT<Double_t>
#pragma link C++ function operator+(const ROOT::Experimental::TEveVectorT<Double_t>&, const ROOT::Experimental::TEveVectorT<Double_t>&);
#pragma link C++ function operator-(const ROOT::Experimental::TEveVectorT<Double_t>&, const ROOT::Experimental::TEveVectorT<Double_t>&);
#pragma link C++ function operator*(const ROOT::Experimental::TEveVectorT<Double_t>&, Double_t);
#pragma link C++ function operator*(Double_t, const ROOT::Experimental::TEveVectorT<Double_t>&);
#pragma link C++ function operator+(const ROOT::Experimental::TEveVector4T<Double_t>&, const ROOT::Experimental::TEveVector4T<Double_t>&);
#pragma link C++ function operator-(const ROOT::Experimental::TEveVector4T<Double_t>&, const ROOT::Experimental::TEveVector4T<Double_t>&);
#pragma link C++ function operator*(const ROOT::Experimental::TEveVector4T<Double_t>&, Double_t);
#pragma link C++ function operator*(Double_t, const ROOT::Experimental::TEveVector4T<Double_t>&);
#pragma link C++ function operator+(const ROOT::Experimental::TEveVector2T<Double_t>&, const ROOT::Experimental::TEveVector2T<Double_t>&);
#pragma link C++ function operator-(const ROOT::Experimental::TEveVector2T<Double_t>&, const ROOT::Experimental::TEveVector2T<Double_t>&);
#pragma link C++ function operator*(const ROOT::Experimental::TEveVector2T<Double_t>&, Double_t);
#pragma link C++ function operator*(Double_t, const ROOT::Experimental::TEveVector2T<Double_t>&);

// TEvePathMark
#pragma link C++ class   ROOT::Experimental::TEvePathMarkT<Float_t>+;
#pragma link C++ class   ROOT::Experimental::TEvePathMarkT<Double_t>+;
#pragma link C++ typedef ROOT::Experimental::TEvePathMark;
#pragma link C++ typedef ROOT::Experimental::TEvePathMarkF;
#pragma link C++ typedef ROOT::Experimental::TEvePathMarkD;

// TEveTrans
#pragma link C++ class ROOT::Experimental::TEveTrans-;

// TEveUtil
#pragma link C++ class ROOT::Experimental::TEveUtil+;
#pragma link C++ class ROOT::Experimental::TEveException+;
#pragma link C++ class ROOT::Experimental::TEvePadHolder+;
#pragma link C++ class ROOT::Experimental::TEveGeoManagerHolder+;
#pragma link C++ class ROOT::Experimental::TEveRefCnt+;
#pragma link C++ class ROOT::Experimental::TEveRefBackPtr+;

// TEveManager
#pragma link C++ class ROOT::Experimental::TEveManager+;
#pragma link C++ global ROOT::Experimental::gEve;
#pragma link C++ class ROOT::Experimental::TEveManager::TRedrawDisabler+;
#pragma link C++ class ROOT::Experimental::TEveManager::TExceptionHandler+;

// TEveVSD
// #pragma link C++ class TEveMCTrack+;
// #pragma link C++ class TEveHit+;
// #pragma link C++ class TEveCluster+;

// #pragma link C++ class   TEveRecTrackT<Float_t>+;
// #pragma link C++ class   TEveRecTrackT<Double_t>+;
// #pragma link C++ typedef TEveRecTrack;
// #pragma link C++ typedef TEveRecTrackF;
// #pragma link C++ typedef TEveRecTrackD;

// #pragma link C++ class TEveRecKink+;
// #pragma link C++ class TEveRecV0+;
// #pragma link C++ class TEveRecCascade+;
// #pragma link C++ class TEveMCRecCrossRef+;

// #pragma link C++ class TEveVSD+;

// TEveChunkManager
// #pragma link C++ class TEveChunkManager+;
// #pragma link C++ class TEveChunkManager::iterator-;

// TEveEventManager
// #pragma link C++ class TEveEventManager+;

// TEveTreeTools
#pragma link C++ class ROOT::Experimental::TEveSelectorToEventList+;
#pragma link C++ class ROOT::Experimental::TEvePointSelectorConsumer+;
#pragma link C++ class ROOT::Experimental::TEvePointSelector+;

// TEveElement
#pragma link C++ class ROOT::Experimental::TEveElement+;
#pragma link C++ class ROOT::Experimental::TEveElementObjectPtr+;
#pragma link C++ class ROOT::Experimental::TEveElementList+;
#pragma link C++ class ROOT::Experimental::TEveElementListProjected+;

#pragma link C++ class std::list<ROOT::Experimental::TEveElement*>;
#pragma link C++ class std::list<ROOT::Experimental::TEveElement*>::iterator;
#pragma link C++ class std::list<ROOT::Experimental::TEveElement*>::const_iterator;
#pragma link C++ typedef ROOT::Experimental::TEveElement::List_t;
#pragma link C++ typedef ROOT::Experimental::TEveElement::List_i;
#pragma link C++ typedef ROOT::Experimental::TEveElement::List_ci;

#pragma link C++ class std::set<ROOT::Experimental::TEveElement*>;
#pragma link C++ class std::set<ROOT::Experimental::TEveElement*>::iterator;
#pragma link C++ class std::set<ROOT::Experimental::TEveElement*>::const_iterator;
#pragma link C++ typedef ROOT::Experimental::TEveElement::Set_t;
#pragma link C++ typedef ROOT::Experimental::TEveElement::Set_i;
#pragma link C++ typedef ROOT::Experimental::TEveElement::Set_ci;

// TEveCompound
#pragma link C++ class ROOT::Experimental::TEveCompound+;
#pragma link C++ class ROOT::Experimental::TEveCompoundProjected+;

// TEveSelection
#pragma link C++ class ROOT::Experimental::TEveSelection+;
#pragma link C++ class ROOT::Experimental::TEveSecondarySelectable+;

// GL-interface
#pragma link C++ class ROOT::Experimental::TEveScene+;
#pragma link C++ class ROOT::Experimental::TEveSceneList+;
#pragma link C++ class ROOT::Experimental::TEveSceneInfo+;
#pragma link C++ class ROOT::Experimental::TEveViewer+;
#pragma link C++ class ROOT::Experimental::TEveViewerList+;
#pragma link C++ class ROOT::Experimental::TEveViewerListEditor+;

// TEvePad
// #pragma link C++ class TEvePad+;

// TEveMacro
#pragma link C++ class ROOT::Experimental::TEveMacro+;

// Projections / non-linear transformations
#pragma link C++ class ROOT::Experimental::TEveProjectable+;
#pragma link C++ class ROOT::Experimental::TEveProjected+;
#pragma link C++ class ROOT::Experimental::TEveProjection+;
#pragma link C++ class ROOT::Experimental::TEveProjection::PreScaleEntry_t+;
#pragma link C++ class std::vector<ROOT::Experimental::TEveProjection::PreScaleEntry_t>;
#pragma link C++ class std::vector<ROOT::Experimental::TEveProjection::PreScaleEntry_t>::iterator;
#pragma link C++ operators std::vector<ROOT::Experimental::TEveProjection::PreScaleEntry_t>::iterator;
#pragma link C++ typedef ROOT::Experimental::TEveProjection::vPreScale_t;
#pragma link C++ typedef ROOT::Experimental::TEveProjection::vPreScale_i;
#pragma link C++ class ROOT::Experimental::TEveRhoZProjection+;
#pragma link C++ class ROOT::Experimental::TEveRPhiProjection+;
#pragma link C++ class ROOT::Experimental::TEve3DProjection+;

#pragma link C++ class ROOT::Experimental::TEveProjectionManager+;
#pragma link C++ class ROOT::Experimental::TEveProjectionAxes+;

// Generic configuration
// #pragma link C++ class TEveParamList;
// #pragma link C++ class TEveParamList::FloatConfig_t+;
// #pragma link C++ class TEveParamList::IntConfig_t+;
// #pragma link C++ class TEveParamList::BoolConfig_t+;
// #pragma link C++ class TEveParamListEditor+;

// TEveTrack
#pragma link C++ class ROOT::Experimental::TEveTrack+;
#pragma link C++ class ROOT::Experimental::TEveTrackList+;
#pragma link C++ class ROOT::Experimental::TEveTrackProjected+;
#pragma link C++ class ROOT::Experimental::TEveTrackListProjected+;

// TEveTrackPropagator
#pragma link C++ class ROOT::Experimental::TEveTrackPropagator+;
#pragma link C++ class ROOT::Experimental::TEveMagField+;
#pragma link C++ class ROOT::Experimental::TEveMagFieldConst+;
#pragma link C++ class ROOT::Experimental::TEveMagFieldDuo+;

// TEvePointSet
#pragma link C++ class ROOT::Experimental::TEvePointSet+;
#pragma link C++ class ROOT::Experimental::TEvePointSetArray+;
#pragma link C++ class ROOT::Experimental::TEvePointSetArrayEditor+;
#pragma link C++ class ROOT::Experimental::TEvePointSetProjected+;

// TEveLine
#pragma link C++ class ROOT::Experimental::TEveLine+;
#pragma link C++ class ROOT::Experimental::TEveLineProjected+;

// Shapes
#pragma link C++ class ROOT::Experimental::TEveShape+;
#pragma link C++ class ROOT::Experimental::TEvePolygonSetProjected+;
#pragma link C++ class ROOT::Experimental::TEveGeoShape+;
#pragma link C++ class ROOT::Experimental::TEveGeoShapeProjected+;
#pragma link C++ class ROOT::Experimental::TEveGeoShapeExtract+;
#pragma link C++ class ROOT::Experimental::TEveGeoPolyShape+;
// Not yet ported
// #pragma link C++ class TEveGeoNode+;
// #pragma link C++ class TEveGeoTopNode+;
