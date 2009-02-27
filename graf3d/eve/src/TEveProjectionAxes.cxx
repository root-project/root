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

#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"
#include "TVirtualPad.h"
#include "TVirtualViewer3D.h"
#include "TEveTrans.h"


// Axes for non-linear projections. Show scale of TEveProjectionManager
// children. With different step mode tick-marks can positioned
// equidistant or placed with value monotonically increasing from lower left corner
// of bounding box.

ClassImp(TEveProjectionAxes);

//______________________________________________________________________________
TEveProjectionAxes::TEveProjectionAxes(TEveProjectionManager* m) :
   TEveElement(fColor),
   TNamed("TEveProjectionAxes", ""),
   fManager(m),

   fColor(kGray),

   fLabMode(kValue),
   fAxesMode(kAll),

   fDrawCenter(kFALSE),
   fDrawOrigin(kFALSE)
{
   // Constructor.

   fCanEditMainTrans = kFALSE;
   fManager->AddDependent(this);

   // Axis attributes.
   fNdivisions = 1010;
   fLabelSize = 0.015;
   fLabelColor = kGray+1;
   fAxisColor = kGray+1;
   fTickLength = 0.015;
   fLabelOffset = 0.01;
}

//______________________________________________________________________________
TEveProjectionAxes::~TEveProjectionAxes()
{
   // Destructor.

   fManager->RemoveDependent(this);
}

//______________________________________________________________________________
void TEveProjectionAxes::Paint(Option_t* )
{
   // Paint this object. Only direct rendering is supported.

   static const TEveException eH("TEveProjectionAxes::Paint ");

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
      Error(eH, "only direct GL rendering supported.");
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
}

//______________________________________________________________________________
const TGPicture* TEveProjectionAxes::GetListTreeIcon(Bool_t)
{
   // Return TEveProjectionAxes icon.

   return TEveElement::fgListTreeIcons[6];
}
