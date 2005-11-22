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

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLManip                                                             //
//                                                                      //
// Abstract base class for viewer manipulators, which allow direct in   //
// viewer manipulation of a (TGlPhysicalShape) object - currently       //
// translation, scaling and rotation along/round objects local axes.    //
// See derived classes for these implementations.                       //
//                                                                      //
// This class provides binding to the zero or one manipulated physical, //
// hit testing (selection) for manipulator sub component (widget), and  //
// some common mouse action handling/tracking.                          //
//////////////////////////////////////////////////////////////////////////

ClassImp(TGLManip)

Float_t TGLManip::fgRed[4]    = {0.8, 0.0, 0.0, 1.0 };
Float_t TGLManip::fgGreen[4]  = {0.0, 0.8, 0.0, 1.0 };
Float_t TGLManip::fgBlue[4]   = {0.0, 0.0, 0.8, 1.0 };
Float_t TGLManip::fgYellow[4] = {0.8, 0.8, 0.0, 1.0 };
Float_t TGLManip::fgWhite[4]  = {1.0, 1.0, 1.0, 1.0 };
Float_t TGLManip::fgGrey[4]   = {0.5, 0.5, 0.5, 0.4 };

//______________________________________________________________________________
TGLManip::TGLManip(TGLViewer & viewer) : 
   fViewer(viewer), fShape(0), 
   fSelectedWidget(0), fActive(kFALSE),
   fFirstMouse(0, 0), 
   fLastMouse(0, 0)
{
   // Construct a manipulator object, bound to supplied viewer, and no physical shape
   
   // TODO: The requirement to attach to viewer is needed for cross thread selection
   // callback under Windows - when the design of TGLKernel / TGLManager is finally
   // resolved this can probably be removed.
}

//______________________________________________________________________________
TGLManip::TGLManip(TGLViewer & viewer, TGLPhysicalShape * shape) : 
   fViewer(viewer), fShape(shape), 
   fSelectedWidget(0), fActive(kFALSE),
   fFirstMouse(0, 0), 
   fLastMouse(0, 0)
{
   // Construct a manipulator object, bound to supplied viewer, and physical shape
   
   // TODO: The requirement to attach to viewer is needed for cross thread selection
   // callback under Windows - when the design of TGLKernel / TGLManager is finally
   // resolved this can probably be removed.
}

//______________________________________________________________________________
TGLManip::~TGLManip() 
{
   // Destroy manipulator object
}

//______________________________________________________________________________
void TGLManip::Select(const TGLCamera & camera) 
{
   // Perform selection (hit testing) to find selected widget (component)
   // of the manipulator - stored in fSelectedWidget
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
   // Handle a mouse button event - return kTRUE if processed, kFALSE otherwise
   
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
   // Handle a mouse button event - return kTRUE if widget selection change
   // kFALSE otherwise
   
   TGLRect selectRect(event->fX, event->fY, 3, 3);
   // Need to do this cross thread under Windows for gVirtualGL context - very ugly...
   // TODO: When the design of TGLKernel / TGLManager is finally resolved this can probably 
   // be removed.
   UInt_t oldSelection = fSelectedWidget;
   fViewer.RequestSelectManip(selectRect);
   return (fSelectedWidget != oldSelection);
}

//______________________________________________________________________________
Double_t TGLManip::CalcDrawScale(const TGLBoundingBox & box, const TGLCamera & camera) const
{
   // Calculates a scale factor (in world units) for drawing manipulators with 
   // reasonable size range in current camera.
   TGLVector3 pixelInWorld = camera.ViewportDeltaToWorld(box.Center(), 1, 1);
   Double_t pixelScale = pixelInWorld.Mag();
   Double_t scale = box.Extents().Mag() / 100.0;
   
   // Allow some variation so zooming is noticable
   if (scale < pixelScale * 3.0) {
      scale = pixelScale * 3.0;
   } else if (scale > pixelScale * 5.0) {
      scale = pixelScale * 5.0;
   }
   return scale;
}
