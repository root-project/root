// @(#)root/gl:$Name:  $:$Id: TGLManip.cxx,v 1.2 2007/05/10 11:17:48 mtadel Exp $
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
#include "TGLPhysicalShape.h"
#include "TGLIncludes.h"
#include "TROOT.h"

#include "TVirtualX.h"

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

Float_t TGLManip::fgRed[4]    = { 0.8, 0.0, 0.0, 1.0 };
Float_t TGLManip::fgGreen[4]  = { 0.0, 0.8, 0.0, 1.0 };
Float_t TGLManip::fgBlue[4]   = { 0.0, 0.0, 0.8, 1.0 };
Float_t TGLManip::fgYellow[4] = { 0.8, 0.8, 0.0, 1.0 };
Float_t TGLManip::fgWhite[4]  = { 1.0, 1.0, 1.0, 1.0 };
Float_t TGLManip::fgGrey[4]   = { 0.5, 0.5, 0.5, 0.4 };

//______________________________________________________________________________
TGLManip::TGLManip() :
   fShape(0),
   fSelectedWidget(0), fActive(kFALSE),
   fFirstMouse(0, 0),
   fLastMouse(0, 0)
{
   // Construct a manipulator object, bound to supplied viewer, and no
   // physical shape.
}

//______________________________________________________________________________
TGLManip::TGLManip(TGLPhysicalShape * shape) :
   fShape(shape),
   fSelectedWidget(0), fActive(kFALSE),
   fFirstMouse(0, 0),
   fLastMouse(0, 0)
{
   // Construct a manipulator object, bound to supplied physical shape.
}

//______________________________________________________________________________
TGLManip::TGLManip(const TGLManip& gm) :
  TVirtualGLManip(gm),
  fShape(gm.fShape),
  fSelectedWidget(gm.fSelectedWidget),
  fActive(gm.fActive),
  fFirstMouse(gm.fFirstMouse),
  fLastMouse(gm.fLastMouse)
{
   // Copy constructor.
}

//______________________________________________________________________________
TGLManip& TGLManip::operator=(const TGLManip& gm)
{
   // Assignement operator.

   if(this!=&gm) {
      TVirtualGLManip::operator=(gm);
      fShape=gm.fShape;
      fSelectedWidget=gm.fSelectedWidget;
      fActive=gm.fActive;
      fFirstMouse=gm.fFirstMouse;
      fLastMouse=gm.fLastMouse;
      for(Int_t i=0; i<4; i++) {
         fgRed[i]=gm.fgRed[i];
         fgGreen[i]=gm.fgGreen[i];
         fgBlue[i]=gm.fgBlue[i];
         fgYellow[i]=gm.fgYellow[i];
         fgWhite[i]=gm.fgWhite[i];
         fgGrey[i]=gm.fgGrey[i];
      }
   }
   return *this;
}

//______________________________________________________________________________
TGLManip::~TGLManip()
{
   // Destroy manipulator object.
}

//______________________________________________________________________________
Bool_t TGLManip::HandleButton(const Event_t & event, const TGLCamera & /*camera*/)
{
   // Handle a mouse button event - return kTRUE if processed, kFALSE otherwise

   // Only interested in Left mouse button actions
   if (event.fCode != kButton1) {
      return kFALSE;
   }

   // Mouse down on selected widget?
   if (event.fType == kButtonPress && fSelectedWidget != 0) {
      fFirstMouse.SetX(event.fX);
      fFirstMouse.SetY(event.fY);
      fLastMouse.SetX(event.fX);
      fLastMouse.SetY(event.fY);
      fActive = kTRUE;
      return kTRUE;
   } else if (event.fType == kButtonRelease && fActive) {
      fActive = kFALSE;
      return kTRUE;
   } else {
      return kFALSE;
   }
}

//______________________________________________________________________________
Bool_t TGLManip::HandleMotion(const Event_t        & /*event*/,
                              const TGLCamera      & /*camera*/)
{
   // Handle a mouse button event - return kTRUE if widget selection change
   // kFALSE otherwise

   return kFALSE;
}

//______________________________________________________________________________
void TGLManip::CalcDrawScale(const TGLBoundingBox & box,
                             const TGLCamera      & camera,
                             Double_t             & base,
                             TGLVector3           axis[3]) const
{
   // Calculates base and axis scale factor (in world units) for
   // drawing manipulators with reasonable size range in current
   // camera.

   // Calculate a base scale
   base = box.Extents().Mag() / 100.0;

   // Clamp this base scale to a viewport pixel range
   // Allow some variation so zooming is noticable
   TGLVector3 pixelInWorld = camera.ViewportDeltaToWorld(box.Center(), 1, 1);
   Double_t pixelScale = pixelInWorld.Mag();
   if (base < pixelScale * 3.0) {
      base = pixelScale * 3.0;
   } else if (base > pixelScale * 6.0) {
      base = pixelScale * 6.0;
   }

   // Calculate some axis scales
   for (UInt_t i = 0; i<3; i++) {
      if (box.IsEmpty()) {
         axis[i] = box.Axis(i, kTRUE)*base*-10.0;
      } else {
         axis[i] = box.Axis(i, kFALSE)*-0.51;
         if (axis[i].Mag() < base*10.0) {
            axis[i] *= base*10.0/axis[i].Mag();
         }
      }
   }
}
