// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveProjectionAxes.h"
#include "TEveProjectionManager.h"
#include "TMath.h"


// Axes for non-linear projections. Show scale of TEveProjectionManager
// children. With different step mode tick-marks can positioned
// equidistant or placed with value monotonically increasing from lower left corner
// of bounding box.

ClassImp(TEveProjectionAxes);

//______________________________________________________________________________
TEveProjectionAxes::TEveProjectionAxes(TEveProjectionManager* m) :
   TEveText("ProjectionAxes"),
   fManager(m),

   fDrawCenter(kFALSE),
   fDrawOrigin(kFALSE),

   fStepMode(kPosition),
   fNumTickMarks(7)
{
   // Constructor.

   SetName("ProjectionAxes");
   fText = "Axes Title";
   fCanEditMainTrans = kFALSE;

   fManager->AddDependent(this);
}

//______________________________________________________________________________
TEveProjectionAxes::~TEveProjectionAxes()
{
   // Destructor.

   fManager->RemoveDependent(this);
}

//______________________________________________________________________________
void TEveProjectionAxes::ComputeBBox()
{
   // Virtual from TAttBBox; fill bounding-box information.

   static const TEveException eH("TEveProjectionManager::ComputeBBox ");

   BBoxZero();
   if(fManager == 0)
      return;

   for (Int_t i=0; i<6; ++i)
      fBBox[i] = fManager->GetBBox()[i];

   AssertBBoxExtents(0.1);
   {
      using namespace TMath;
      fBBox[0] = 10.0f * Floor(fBBox[0]/10.0f);
      fBBox[1] = 10.0f * Ceil (fBBox[1]/10.0f);
      fBBox[2] = 10.0f * Floor(fBBox[2]/10.0f);
      fBBox[3] = 10.0f * Ceil (fBBox[3]/10.0f);
   }
}

//______________________________________________________________________________
const TGPicture* TEveProjectionAxes::GetListTreeIcon(Bool_t)
{
   // Return TEveProjectionAxes icon.

   return TEveElement::fgListTreeIcons[6];
}
