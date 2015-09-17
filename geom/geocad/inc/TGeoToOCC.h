// @(#)geom/geocad:$Id$
// Author: Cinzia Luzzi   5/5/2012

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoToOCC
#define ROOT_TGeoToOCC

//Cascade
#include <Standard_Version.hxx>

#define Printf Printf_opencascade
#include <TopoDS_Shape.hxx>
#include <TopoDS_Wire.hxx>
#undef Printf

//Root
#ifndef ROOT_TGeoXtru
#include "TGeoXtru.h"
#endif
#ifndef ROOT_TGeoCompositeShape
#include "TGeoCompositeShape.h"
#endif

#include <fstream>


class TGeoToOCC
{
private:
   void OCCDocCreation();
   TopoDS_Shape OCC_Arb8(Double_t dz, Double_t * ivert, Double_t * points);
   TopoDS_Shape OCC_EllTube(Double_t Dx, Double_t Dy, Double_t Dz);
   TopoDS_Shape OCC_Torus(Double_t Rmin, Double_t Rmax, Double_t Rtor, Double_t SPhi, Double_t DPhi);
   TopoDS_Shape OCC_Sphere(Double_t rmin, Double_t rmax, Double_t phi1, Double_t Dphi, Double_t theta1, Double_t Dtheta);
   TopoDS_Shape OCC_Tube(Double_t rmin, Double_t rmax, Double_t dz, Double_t phi1, Double_t phi2);
   TopoDS_Shape OCC_Cones(Double_t rmin1, Double_t rmax1, Double_t rmin2, Double_t rmax2, Double_t dz, Double_t phi1, Double_t phi2);
   TopoDS_Shape OCC_Cuttub(Double_t rmin, Double_t rmax, Double_t dz, Double_t phi1, Double_t Dphi,const Double_t * Nlow,const Double_t * Nhigh);
   TopoDS_Shape OCC_Hype(Double_t rmin, Double_t  rmax,Double_t  stin, Double_t stout, Double_t  dz );
   TopoDS_Wire Polygon(Double_t *x, Double_t *y, Double_t z, Int_t num );
   TopoDS_Shape OCC_ParaTrap (Double_t *vertex);
   TopoDS_Shape Gtra_Arb8Creation(Double_t *vertex, Int_t *faces, Int_t fNumber);
   TopoDS_Shape OCC_Pcon(Double_t startPhi, Double_t deltaPhi,Int_t zNum, Double_t *rMin, Double_t *rMax, Double_t *z);
   TopoDS_Shape OCC_Xtru(TGeoXtru * TG_Xtru);
   TopoDS_Shape OCC_Pgon(Int_t np, Int_t nz, Double_t * p, Double_t phi1, Double_t DPhi, Int_t numpoint);
   TopoDS_Shape OCC_Box(Double_t dx, Double_t dy, Double_t dz, Double_t OX, Double_t OY, Double_t OZ);
   TopoDS_Shape OCC_Trd(Double_t dx1, Double_t dx2, Double_t dy1, Double_t dy2, Double_t dz);
   ofstream out;
   TopoDS_Shape fOccShape;

public:
   TGeoToOCC();
   virtual ~TGeoToOCC();
   TopoDS_Shape OCC_SimpleShape(TGeoShape *TG_Shape);
   TopoDS_Shape OCC_CompositeShape(TGeoCompositeShape *cs, TGeoHMatrix matrix);
   TopoDS_Shape Reverse(TopoDS_Shape Shape);

};
#endif


