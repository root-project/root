// @(#)root/geom:$Name:  $:$Id: LinkDef.h,v 1.2 2002/07/10 19:24:16 brun Exp $
// Author : Andrei Gheata 10/06/02
/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ global gGeoManager;
#pragma link C++ global gGeoIdentity;
#pragma link C++ global gGeoMatrixCache;

#pragma link C++ class TGeoAtt+;
#pragma link C++ class TGeoBoolCombinator+;
#pragma link C++ class TGeoMaterial+;
#pragma link C++ class TGeoMixture+;
#pragma link C++ class TGeoMatrix+;
#pragma link C++ class TGeoHMatrix+;
#pragma link C++ class TGeoTranslation+;
#pragma link C++ class TGeoRotation+;
#pragma link C++ class TGeoCombiTrans+;
#pragma link C++ class TGeoGenTrans+;
#pragma link C++ class TGeoScale+;
#pragma link C++ class TGeoIdentity+;
#pragma link C++ class TGeoFinder+;
#pragma link C++ class TGeoVoxelFinder+;
#pragma link C++ class TGeoPatternFinder+;
#pragma link C++ class TGeoPatternX+;
#pragma link C++ class TGeoPatternY+;
#pragma link C++ class TGeoPatternZ+;
#pragma link C++ class TGeoPatternParaX+;
#pragma link C++ class TGeoPatternParaY+;
#pragma link C++ class TGeoPatternParaZ+;
#pragma link C++ class TGeoPatternTrapZ+;
#pragma link C++ class TGeoPatternCylR+;
#pragma link C++ class TGeoPatternCylPhi+;
#pragma link C++ class TGeoPatternSphR+;
#pragma link C++ class TGeoPatternSphTheta+;
#pragma link C++ class TGeoPatternSphPhi+;
#pragma link C++ class TGeoPatternHoneycomb+;
#pragma link C++ class TGeoParamCurve+;
#pragma link C++ class TGeoShape+;
#pragma link C++ class TGeoBBox+;
#pragma link C++ class TGeoPara+;
#pragma link C++ class TGeoSphere+;
#pragma link C++ class TGeoTube+;
#pragma link C++ class TGeoTubeSeg+;
#pragma link C++ class TGeoCtub+;
#pragma link C++ class TGeoEltu+;
#pragma link C++ class TGeoCone+;
#pragma link C++ class TGeoConeSeg+;
#pragma link C++ class TGeoPcon+;
#pragma link C++ class TGeoPgon+;
#pragma link C++ class TGeoArb8+;
#pragma link C++ class TGeoTrap+;
#pragma link C++ class TGeoGtra+;
#pragma link C++ class TGeoTrd1+;
#pragma link C++ class TGeoTrd2+;
#pragma link C++ class TGeoCompositeShape+;
#pragma link C++ class TGeoVolume+;
#pragma link C++ class TGeoVolumeMulti+;
#pragma link C++ class TGeoNode+;
#pragma link C++ class TGeoNodeMatrix+;
#pragma link C++ class TGeoNodeOffset+;
#pragma link C++ class TGeoManager+;
#pragma link C++ class TGeoNodeArray+;
#pragma link C++ class TGeoNodeObjArray+;
#pragma link C++ class TGeoNodeCache+;
#pragma link C++ class TGeoCacheDummy+;
#pragma link C++ class TGeoMatrixCache+;
#pragma link C++ class TGeoNodePos+;
#pragma link C++ class TGeoCacheState+;
#pragma link C++ class TGeoCacheStateDummy+;
#pragma link C++ class TGeoMatHandler+;
#pragma link C++ class TGeoMatHandlerId+;
#pragma link C++ class TGeoMatHandlerX+;
#pragma link C++ class TGeoMatHandlerY+;
#pragma link C++ class TGeoMatHandlerZ+;
#pragma link C++ class TGeoMatHandlerXY+;
#pragma link C++ class TGeoMatHandlerXZ+;
#pragma link C++ class TGeoMatHandlerYZ+;
#pragma link C++ class TGeoMatHandlerXYZ+;
#pragma link C++ class TGeoMatHandlerRot+;
#pragma link C++ class TGeoMatHandlerRotTr+;
#pragma link C++ class TGeoMatHandlerScl+;
#pragma link C++ class TGeoMatHandlerTrScl+;
#pragma link C++ class TGeoMatHandlerRotScl+;
#pragma link C++ class TGeoMatHandlerRotTrScl+;
#pragma link C++ class TVirtualGeoPainter+;


#endif
