// @(#)root/gl:$Name:  $:$Id: TGLManip.cxx
// Author:  Richard Maunder  16/09/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLManip.h"
#include "TGLUtil.h"
#include "TGLCamera.h"
#include "TGLViewer.h"
#include "TGLPhysicalShape.h"
#include "TGLIncludes.h"

Float_t TGLManip::fgRed[4]    = {0.8, 0.0, 0.0, 1.0 };
Float_t TGLManip::fgGreen[4]  = {0.0, 0.8, 0.0, 1.0 };
Float_t TGLManip::fgBlue[4]   = {0.0, 0.0, 0.8, 1.0 };
Float_t TGLManip::fgYellow[4] = {0.8, 0.8, 0.0, 1.0 };
Float_t TGLManip::fgWhite[4]  = {1.0, 1.0, 1.0, 1.0 };
Float_t TGLManip::fgGrey[4]   = {0.5, 0.5, 0.5, 0.4 };

ClassImp(TGLManip)

//______________________________________________________________________________
TGLManip::TGLManip(TGLViewer & viewer) : 
   fViewer(viewer), fShape(0), 
   fSelectedWidget(0), fActive(kFALSE),
   fFirstMouse(0, 0), 
   fLastMouse(0, 0)
{
}

//______________________________________________________________________________
TGLManip::TGLManip(TGLViewer & viewer, TGLPhysicalShape * shape) : 
   fViewer(viewer), fShape(shape), 
   fSelectedWidget(0), fActive(kFALSE),
   fFirstMouse(0, 0), 
   fLastMouse(0, 0)
{
}

//______________________________________________________________________________
TGLManip::~TGLManip() 
{
}

//______________________________________________________________________________
void TGLManip::Select(const TGLCamera & camera) 
{
   static UInt_t selectBuffer[4*4];
   glSelectBuffer(4*4, &selectBuffer[0]);
   glRenderMode(GL_SELECT);
   glInitNames();
   Draw(camera);
   Int_t hits = glRenderMode(GL_RENDER);
   TGLUtil::CheckError();
   if (hits < 0) {
      Error("TGLManip::Select", "selection buffer overflow");
      return;
   }

   if (hits > 0) {
      fSelectedWidget = 0;
      UInt_t minDepth = kMaxUInt;
      for (Int_t i = 0; i < hits; i++) {
         // Skip selection on unnamed hits
         if (selectBuffer[i * 4] == 0) {
            continue;
         }
         if (selectBuffer[i * 4 + 1] < minDepth) {
            fSelectedWidget = selectBuffer[i * 4 + 3];
            minDepth = selectBuffer[i * 4 + 1];
         }
      }
   } else {
      fSelectedWidget = 0;
   }
}

//______________________________________________________________________________
Bool_t TGLManip::HandleButton(const Event_t * event, const TGLCamera & /*camera*/)
{
   // Only interested in Left mouse button actions
   if (event->fCode != kButton1) {
      return kFALSE;
   }

   // Mouse down on selected widget?
   if (event->fType == kButtonPress && fSelectedWidget != 0) {
      fFirstMouse.SetX(event->fX);
      fFirstMouse.SetY(event->fY);
      fLastMouse.SetX(event->fX);
      fLastMouse.SetY(event->fY);
      fActive = kTRUE;
      return kTRUE;
   } else if (event->fType == kButtonRelease && fActive) {
      fActive = kFALSE;
      return kTRUE;
   } else {
      return kFALSE;
   }
}

//______________________________________________________________________________
Bool_t TGLManip::HandleMotion(const Event_t * event, const TGLCamera & /*camera*/)
{
   TGLRect selectRect(event->fX, event->fY, 3, 3);
   // Need to do this cross thread under Windows for gVirtualGL context - very ugly...
   UInt_t oldSelection = fSelectedWidget;
   fViewer.RequestSelectManip(selectRect);
   return (fSelectedWidget != oldSelection);
}

//______________________________________________________________________________
Double_t TGLManip::CalcDrawScale(const TGLBoundingBox & box, const TGLCamera & camera) const
{
   TGLVector3 pixelInWorld = camera.ViewportDeltaToWorld(box.Center(), 1, 1);
   Double_t pixelScale = pixelInWorld.Mag();
   Double_t scale = box.Extents().Mag() / 100.0;
   if (scale < pixelScale * 3.0) {
      scale = pixelScale * 3.0;
   } else if (scale > pixelScale * 5.0) {
      scale = pixelScale * 5.0;
   }
   return scale;
}
