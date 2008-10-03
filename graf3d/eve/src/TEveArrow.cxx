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

#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"
#include "TVirtualPad.h"
#include "TVirtualViewer3D.h"


// TEveElement class used for display of arrow.

ClassImp(TEveArrow);

//______________________________________________________________________________
TEveArrow::TEveArrow(Float_t xVec, Float_t yVec, Float_t zVec,
                     Float_t x0,   Float_t y0,   Float_t z0):
   TEveElement(fColor),
   TNamed("TEveArrow", ""),
   TAtt3D(),
   TAttBBox(),

   fTubeR(0.02),
   fConeR(0.04),
   fConeL(0.08)
{
   // Constructor.

   fVector.Set(xVec, yVec, zVec);
   fOrigin.Set(x0, y0, z0);
}

//______________________________________________________________________________
void TEveArrow::ComputeBBox()
{
   // Compute bounding-box of the data.

   BBoxZero();
   BBoxCheckPoint(fOrigin.fX, fOrigin.fY, fOrigin.fZ);
   BBoxCheckPoint(fOrigin.fX+fVector.fX, fOrigin.fY+fVector.fY, fOrigin.fZ+fVector.fZ);
}

//______________________________________________________________________________
void TEveArrow::Paint(Option_t* /*option*/)
{
   // Paint object.
   // This is for direct rendering (using TEveArrowGL class).

   static const TEveException eh("TEveArrow::Paint ");

   if (fRnrSelf == kFALSE) return;

   TBuffer3D buff(TBuffer3DTypes::kGeneric);

   // Section kCore
   buff.fID           = this;
   buff.fColor        = GetMainColor();
   buff.fTransparency = GetMainTransparency();
   if (HasMainTrans())
      RefMainTrans().SetBuffer3D(buff);
   buff.SetSectionsValid(TBuffer3D::kCore);

   Int_t reqSections = gPad->GetViewer3D()->AddObject(buff);
   if (reqSections != TBuffer3D::kNone)
      Error(eh, "only direct GL rendering supported.");
}
