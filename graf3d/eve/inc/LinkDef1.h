// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006 - 2009

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

//==============================================================================
// LinkDef1.h - Core EVE objects and services.
//==============================================================================

// TEveVector
#pragma link C++ class   TEveVectorT<Float_t>+;
#pragma link C++ class   TEveVectorT<Double_t>+;
#pragma link C++ typedef TEveVector;
#pragma link C++ typedef TEveVectorF;
#pragma link C++ typedef TEveVectorD;

#pragma link C++ class   TEveVector4T<Float_t>+;
#pragma link C++ class   TEveVector4T<Double_t>+;
#pragma link C++ typedef TEveVector4;
#pragma link C++ typedef TEveVector4F;
#pragma link C++ typedef TEveVector4D;

#pragma link C++ class   TEveVector2T<Float_t>+;
#pragma link C++ class   TEveVector2T<Double_t>+;
#pragma link C++ typedef TEveVector2;
#pragma link C++ typedef TEveVector2F;
#pragma link C++ typedef TEveVector2D;

// Operators for TEveVectorXT<Float_t>
#pragma link C++ function operator+(const TEveVectorT<Float_t>&, const TEveVectorT<Float_t>&);
#pragma link C++ function operator-(const TEveVectorT<Float_t>&, const TEveVectorT<Float_t>&);
#pragma link C++ function operator*(const TEveVectorT<Float_t>&, Float_t);
#pragma link C++ function operator*(Float_t, const TEveVectorT<Float_t>&);
#pragma link C++ function operator+(const TEveVector4T<Float_t>&, const TEveVector4T<Float_t>&);
#pragma link C++ function operator-(const TEveVector4T<Float_t>&, const TEveVector4T<Float_t>&);
#pragma link C++ function operator*(const TEveVector4T<Float_t>&, Float_t);
#pragma link C++ function operator*(Float_t, const TEveVector4T<Float_t>&);
#pragma link C++ function operator+(const TEveVector2T<Float_t>&, const TEveVector2T<Float_t>&);
#pragma link C++ function operator-(const TEveVector2T<Float_t>&, const TEveVector2T<Float_t>&);
#pragma link C++ function operator*(const TEveVector2T<Float_t>&, Float_t);
#pragma link C++ function operator*(Float_t, const TEveVector2T<Float_t>&);
// Operators for TEveVectorXT<Double_t>
#pragma link C++ function operator+(const TEveVectorT<Double_t>&, const TEveVectorT<Double_t>&);
#pragma link C++ function operator-(const TEveVectorT<Double_t>&, const TEveVectorT<Double_t>&);
#pragma link C++ function operator*(const TEveVectorT<Double_t>&, Double_t);
#pragma link C++ function operator*(Double_t, const TEveVectorT<Double_t>&);
#pragma link C++ function operator+(const TEveVector4T<Double_t>&, const TEveVector4T<Double_t>&);
#pragma link C++ function operator-(const TEveVector4T<Double_t>&, const TEveVector4T<Double_t>&);
#pragma link C++ function operator*(const TEveVector4T<Double_t>&, Double_t);
#pragma link C++ function operator*(Double_t, const TEveVector4T<Double_t>&);
#pragma link C++ function operator+(const TEveVector2T<Double_t>&, const TEveVector2T<Double_t>&);
#pragma link C++ function operator-(const TEveVector2T<Double_t>&, const TEveVector2T<Double_t>&);
#pragma link C++ function operator*(const TEveVector2T<Double_t>&, Double_t);
#pragma link C++ function operator*(Double_t, const TEveVector2T<Double_t>&);

// TEvePathMark
#pragma link C++ class   TEvePathMarkT<Float_t>+;
#pragma link C++ class   TEvePathMarkT<Double_t>+;
#pragma link C++ typedef TEvePathMark;
#pragma link C++ typedef TEvePathMarkF;
#pragma link C++ typedef TEvePathMarkD;

// TEveTrans
#pragma link C++ class TEveTrans-;
#pragma link C++ class TEveTransSubEditor+;
#pragma link C++ class TEveTransEditor+;

// TEveUtil
#pragma link C++ class TEveUtil+;
#pragma link C++ class TEveException+;
#pragma link C++ class TEvePadHolder+;
#pragma link C++ class TEveGeoManagerHolder+;
#pragma link C++ class TEveRefCnt+;
#pragma link C++ class TEveRefBackPtr+;

// TEveManager
#pragma link C++ class TEveManager+;
#pragma link C++ global gEve;
#pragma link C++ class TEveManager::TRedrawDisabler+;
#pragma link C++ class TEveManager::TExceptionHandler+;

// TEveVSD
#pragma link C++ class TEveVSD+;
#pragma link C++ class TEveMCTrack+;
#pragma link C++ class TEveHit+;
#pragma link C++ class TEveCluster+;
#pragma link C++ class TEveRecTrack+;
#pragma link C++ class TEveRecKink+;
#pragma link C++ class TEveRecV0+;
#pragma link C++ class TEveRecCascade+;
#pragma link C++ class TEveMCRecCrossRef+;

// TEveChunkManager
#pragma link C++ class TEveChunkManager+;
#pragma link C++ class TEveChunkManager::iterator-;

// TEveEventManager
#pragma link C++ class TEveEventManager+;

// TEveTreeTools
#pragma link C++ class TEveSelectorToEventList+;
#pragma link C++ class TEvePointSelectorConsumer+;
#pragma link C++ class TEvePointSelector+;

// TEveElement
#pragma link C++ class TEveElement+;
#pragma link C++ class TEveElement::TEveListTreeInfo+;
#pragma link C++ class TEveElementObjectPtr+;
#pragma link C++ class TEveElementList+;
#pragma link C++ class TEveElementListProjected+;
#pragma link C++ class TEveElementEditor+;

#pragma link C++ class std::list<TEveElement*>;
#pragma link C++ class std::list<TEveElement*>::iterator;
#pragma link C++ class std::list<TEveElement*>::const_iterator;
#pragma link C++ typedef TEveElement::List_t;
#pragma link C++ typedef TEveElement::List_i;
#pragma link C++ typedef TEveElement::List_ci;

#pragma link C++ class std::set<TEveElement*>;
#pragma link C++ class std::set<TEveElement*>::iterator;
#pragma link C++ class std::set<TEveElement*>::const_iterator;
#pragma link C++ typedef TEveElement::Set_t;
#pragma link C++ typedef TEveElement::Set_i;
#pragma link C++ typedef TEveElement::Set_ci;

// TEveCompound
#pragma link C++ class TEveCompound+;
#pragma link C++ class TEveCompoundProjected+;

// TEveSelection
#pragma link C++ class TEveSelection+;
#pragma link C++ class TEveSecondarySelectable+;

// GL-interface
#pragma link C++ class TEveScene+;
#pragma link C++ class TEveSceneList+;
#pragma link C++ class TEveSceneInfo+;
#pragma link C++ class TEveViewer+;
#pragma link C++ class TEveViewerList+;
#pragma link C++ class TEveViewerListEditor+;

// TEvePad
#pragma link C++ class TEvePad+;

// TEveBrowser, TEveCompositeFrame, TEveWindow, TEveWindowManager
#pragma link C++ class TEveListTreeItem+;
#pragma link C++ class TEveGListTreeEditorFrame+;
#pragma link C++ class TEveBrowser+;
#pragma link C++ class TEveCompositeFrame+;
#pragma link C++ class TEveCompositeFrameInMainFrame+;
#pragma link C++ class TEveCompositeFrameInPack+;
#pragma link C++ class TEveCompositeFrameInTab+;
#pragma link C++ class TEveWindow+;
#pragma link C++ class TEveWindowEditor+;
#pragma link C++ class TEveWindowSlot+;
#pragma link C++ class TEveWindowFrame+;
#pragma link C++ class TEveWindowPack+;
#pragma link C++ class TEveWindowTab+;
#pragma link C++ class TEveWindowManager+;

// TEveGedEditor
#pragma link C++ class TEveGedEditor+;
#pragma link C++ class TEveGedNameFrame+;
#pragma link C++ class TEveGedNameTextButton+;

// TEveMacro
#pragma link C++ class TEveMacro+;

// RGValuators
#pragma link C++ class TEveGValuatorBase+;
#pragma link C++ class TEveGValuator+;
#pragma link C++ class TEveGDoubleValuator+;
#pragma link C++ class TEveGTriVecValuator+;

// Projections / non-linear transformations
#pragma link C++ class TEveProjectable+;
#pragma link C++ class TEveProjected+;
#pragma link C++ class TEveProjection+;
#pragma link C++ class TEveProjection::PreScaleEntry_t+;
#pragma link C++ class std::vector<TEveProjection::PreScaleEntry_t>;
#pragma link C++ class std::vector<TEveProjection::PreScaleEntry_t>::iterator;
#pragma link C++ operators std::vector<TEveProjection::PreScaleEntry_t>::iterator;
#pragma link C++ typedef TEveProjection::vPreScale_t;
#pragma link C++ typedef TEveProjection::vPreScale_i;
#pragma link C++ class TEveRhoZProjection+;
#pragma link C++ class TEveRPhiProjection+;
#pragma link C++ class TEve3DProjection+;

#pragma link C++ class TEveProjectionManager+;
#pragma link C++ class TEveProjectionManagerEditor+;
#pragma link C++ class TEveProjectionAxes+;
#pragma link C++ class TEveProjectionAxesEditor+;
#pragma link C++ class TEveProjectionAxesGL+;

// Generic configuration
#pragma link C++ class TEveParamList;
#pragma link C++ class TEveParamList::FloatConfig_t+;
#pragma link C++ class TEveParamList::IntConfig_t+;
#pragma link C++ class TEveParamList::BoolConfig_t+;
#pragma link C++ class TEveParamListEditor+;
