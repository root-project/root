// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveArrow.h"
#include "TEveTrans.h"


//______________________________________________________________________________
//
// TEveElement class used for display of a thick arrow.

ClassImp(TEveArrow);

//______________________________________________________________________________
TEveArrow::TEveArrow(Float_t xVec, Float_t yVec, Float_t zVec,
                     Float_t xOrg, Float_t yOrg, Float_t zOrg):
   TEveElement(fColor),
   TNamed("TEveArrow", ""),
   TAtt3D(), TAttBBox(),

   fTubeR(0.02), fConeR(0.04), fConeL(0.08),
   fOrigin(xOrg, yOrg, zOrg), fVector(xVec, yVec, zVec),
   fDrawQuality(10)
{
   // Constructor.
   // Org - starting point.
   // Vec - vector from start to end of the arrow.

   fCanEditMainColor        = kTRUE;
   fCanEditMainTransparency = kTRUE;
}

//______________________________________________________________________________
void TEveArrow::ComputeBBox()
{
   // Compute bounding-box of the arrow.

   TEveVector a, b;
   fVector.OrthoNormBase(a, b);
   Float_t r = TMath::Max(fTubeR, fConeR);
   a *= r; b *= r;

   TEveVector end(fOrigin + fVector);

   BBoxZero();
   BBoxCheckPoint(fOrigin + a + b);
   BBoxCheckPoint(fOrigin + a - b);
   BBoxCheckPoint(fOrigin - a - b);
   BBoxCheckPoint(fOrigin - a + b);
   BBoxCheckPoint(end + a + b);
   BBoxCheckPoint(end + a - b);
   BBoxCheckPoint(end - a - b);
   BBoxCheckPoint(end - a + b);
}

//______________________________________________________________________________
void TEveArrow::Paint(Option_t*)
{
   // Paint object.
   // This is for direct rendering (using TEveArrowGL class).

   PaintStandard(this);
}
