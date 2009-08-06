// @(#)root/eve:$Id$
// Author: Matevz Tadel, 2009

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEvePlot3D.h"
#include "TEveTrans.h"

#include "TVirtualPad.h"
#include "TVirtualViewer3D.h"
#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"


//______________________________________________________________________________
// Description of TEvePlot3D
//

ClassImp(TEvePlot3D);

//______________________________________________________________________________
TEvePlot3D::TEvePlot3D(const char* n, const char* t) :
   TEveElementList(n, t),
   fPlot(0),
   fLogX(kFALSE), fLogY(kFALSE), fLogZ(kFALSE)
{
   // Constructor.

   InitMainTrans();
}


/******************************************************************************/

/*
// For now use true sizes of plots.
//______________________________________________________________________________
void TEvePlot3D::ComputeBBox()
{
   // Compute bounding-box of the data.

   if (fPlot)
      BBoxZero(); // should be BBoxZero(0.5); once the plots are stuffed into unit box.
   else
      BBoxZero();
}
*/

//______________________________________________________________________________
void TEvePlot3D::Paint(Option_t*)
{
   // Paint object.
   // This is for direct rendering (using TEvePlot3DGL class).

   static const TEveException eh("TEvePlot3D::Paint ");

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
