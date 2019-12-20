// @(#)root/geom:$Id$
// Author : Andrei Gheata 10/06/02
/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifdef __CINT__

#pragma link C++ global gGeoManager;
#pragma link C++ global gGeoIdentity;

#pragma link C++ class TGeoAtt+;
#pragma link C++ struct TGeoStateInfo+;
#pragma link C++ class TGeoBoolNode+;
#pragma link C++ class TGeoUnion+;
#pragma link C++ class TGeoIntersection+;
#pragma link C++ class TGeoSubtraction+;
#pragma link C++ class TGeoMedium+;
#pragma link C++ class TGeoOpticalSurface+;
#pragma link C++ enum  TGeoOpticalSurface::ESurfaceType;
#pragma link C++ enum  TGeoOpticalSurface::ESurfaceModel;
#pragma link C++ enum  TGeoOpticalSurface::ESurfaceFinish;
#pragma link C++ class TGeoSkinSurface+;
#pragma link C++ class TGeoBorderSurface+;
#pragma link C++ class TGeoElement+;
#pragma read sourceClass="TGeoElement" targetClass="TGeoElement" version="[1-2]" source="" target="" \
    code="{ newObj->ComputeDerivedQuantities() ; }" 
#pragma link C++ class TGeoElementRN+;
#pragma link C++ class TGeoIsotope+;
#pragma link C++ class TGeoDecayChannel+;
#pragma link C++ class TGeoElemIter+;
#pragma link C++ class TGeoBatemanSol+;
#pragma link C++ class TGeoElementTable+;
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
#pragma link C++ class TGeoVoxelFinder-;
#pragma link C++ class TGeoShape+;
#pragma link C++ class TGeoHelix+;
#pragma link C++ class TGeoHalfSpace+;
#pragma link C++ class TGeoBBox+;
#pragma link C++ class TGeoPara+;
#pragma link C++ class TGeoSphere+;
#pragma link C++ class TGeoTube+;
#pragma link C++ class TGeoTorus+;
#pragma link C++ class TGeoTubeSeg+;
#pragma read sourceClass="TGeoTubeSeg" targetClass="TGeoTubeSeg" version="[1]" source="" target="" \
    code="{ newObj->AfterStreamer() ; }" 
#pragma link C++ class TGeoCtub+;
#pragma link C++ class TGeoEltu+;
#pragma link C++ class TGeoHype+;
#pragma link C++ class TGeoCone+;
#pragma link C++ class TGeoConeSeg+;
#pragma read sourceClass="TGeoConeSeg" targetClass="TGeoConeSeg" version="[1]" source="" target="" \
    code="{ newObj->AfterStreamer() ; }" 
#pragma link C++ class TGeoParaboloid+;
#pragma link C++ class TGeoPcon-;
#pragma link C++ class TGeoPgon+;
#pragma link C++ class TGeoArb8-;
#pragma link C++ class TGeoTrap+;
#pragma link C++ class TGeoGtra+;
#pragma link C++ class TGeoTrd1+;
#pragma link C++ class TGeoTrd2+;
#pragma link C++ class TGeoCompositeShape+;
#pragma link C++ class TGeoPolygon+;
#pragma link C++ class TGeoXtru+;
#pragma link C++ struct TGeoVector3+;
#pragma link C++ struct TGeoFacet+;
#pragma link C++ class TGeoTessellated+;
#pragma link C++ class TGeoShapeAssembly+;
#pragma link C++ class TGeoScaledShape+;
#pragma link C++ class TGeoVolume-;
#pragma link C++ class TGeoVolumeAssembly+;
#pragma link C++ class TGeoVolumeMulti+;
#pragma link C++ class TGeoNode+;
#pragma link C++ class TGeoPhysicalNode+;
#pragma link C++ class TGeoPNEntry+;
#pragma link C++ class TGeoNodeMatrix+;
#pragma link C++ class TGeoNodeOffset+;
#pragma link C++ class TGeoManager-;
#pragma link C++ class TGeoRegionCut+;
#pragma link C++ class TGeoRegion+;
#pragma link C++ class TVirtualGeoPainter+;
#pragma link C++ class TVirtualGeoTrack+;
#pragma link C++ class TVirtualGeoConverter+;
#pragma link C++ class TGeoIterator;
#pragma link C++ class TGeoIteratorPlugin;
#pragma link C++ class TGeoBuilder;
#pragma link C++ class TGeoNavigator+;
#pragma link C++ class TGeoNavigatorArray;
#pragma link C++ class TGDMLMatrix+;
#pragma link C++ struct std::map<std::thread::id, TGeoNavigatorArray *>;
#pragma link C++ struct std::pair<std::thread::id, TGeoNavigatorArray *>;
#pragma link C++ struct std::map<std::thread::id, Int_t>;
#pragma link C++ struct std::pair<std::thread::id, Int_t>;

#endif
