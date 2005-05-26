// @(#)root/gl:$Name:$:$Id:$
// Author:  Richard Maunder  25/05/2005
// Parts taken from original by Timur Pocheptsov

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// TODO: Function descriptions
// TODO: Class def - same as header

#include "TGLPerspectiveCamera.h"
#include "TGLUtil.h"
#include "TGLIncludes.h"

#include "TMath.h"
#include "Riostream.h"

// TODO: Find somewhere else
#define PI 3.141592654

ClassImp(TGLPerspectiveCamera)

Double_t TGLPerspectiveCamera::fgFOVMin = 5;
Double_t TGLPerspectiveCamera::fgFOVDefault = 30;
Double_t TGLPerspectiveCamera::fgFOVMax = 160;

UInt_t   TGLPerspectiveCamera::fgDollyDeltaSens = 1000;
UInt_t   TGLPerspectiveCamera::fgFOVDeltaSens = 1000;

//______________________________________________________________________________
TGLPerspectiveCamera::TGLPerspectiveCamera()
{
   Setup(TGLBoundingBox(TGLVertex3(-100,-100,-100), TGLVertex3(100,100,100)));
}


//______________________________________________________________________________
TGLPerspectiveCamera::~TGLPerspectiveCamera()
{
}

//______________________________________________________________________________
void TGLPerspectiveCamera::Setup(const TGLBoundingBox & box)
{
   // Setup camera limits based on supplied bounding box.

   fCenter = box.Center();

   // At default FOV, the maximum dolly should just encapsulate the longest side of box with
   // offset for next longetst side

   // Find two longest sides
   TGLVector3 extents = box.Extents();
   Int_t sortInd[3];
   TMath::Sort(3, extents.CArr(), sortInd);
   Double_t longest = extents[sortInd[0]];
   Double_t nextLongest = extents[sortInd[1]];

   fDollyDefault = longest/(2.0*tan(fFOV*PI/360.0));
   fDollyDefault += nextLongest/2.0;
   fDollyDefault *= 1.5;
   fDollyMax = fDollyDefault * 2.0;
   fDollyMin = 0.0;

   fVolumeDiag = box.Extents().Mag();

   Reset();
}

//______________________________________________________________________________
void TGLPerspectiveCamera::Reset()
{
   fFOV = fgFOVDefault;
   fHRotate = 0.0;
   fVRotate = -90.0;
   fTruck.Set(-fCenter.X(), -fCenter.Y(), -fCenter.Z());
   fDolly = fDollyDefault;
   fCacheDirty = kTRUE;
}

//______________________________________________________________________________
Bool_t TGLPerspectiveCamera::Zoom(Int_t shift)
{
   if (AdjustAndClampVal(fFOV, fgFOVMin, fgFOVMax, shift, fgFOVDeltaSens))
   {
      fCacheDirty = kTRUE;
      return kTRUE;
   }
   else
   {
      return kFALSE;
   }
}

//______________________________________________________________________________
Bool_t TGLPerspectiveCamera::Dolly(Int_t shift)
{
   if (AdjustAndClampVal(fDolly, fDollyMin, fDollyMax, shift, fgDollyDeltaSens))
   {
      fCacheDirty = kTRUE;
      return kTRUE;
   }
   else
   {
      return kFALSE;
   }
}

//______________________________________________________________________________
Bool_t TGLPerspectiveCamera::Truck(Int_t x, Int_t y, Int_t xDelta, Int_t yDelta)
{
   //TODO: Convert TGLRect so this not required
   GLint viewport[4] = { fViewport.X(), fViewport.Y(), fViewport.Width(), fViewport.Height() };
   TGLVertex3 start, end;
   gluUnProject(x, y, 1.0, fModVM.CArr(), fProjM.CArr(), viewport, &start.X(), &start.Y(), &start.Z());
   gluUnProject(x + xDelta, y + yDelta, 1.0, fModVM.CArr(), fProjM.CArr(), viewport, &end.X(), &end.Y(), &end.Z());
   TGLVector3 truckDelta = end - start;

   // TODO: Work out correct scaling for this!
   truckDelta /= 2.0;
   fTruck = fTruck + truckDelta;
   fCacheDirty = kTRUE;
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLPerspectiveCamera::Rotate(Int_t xShift, Int_t yShift)
{
   fHRotate += static_cast<float>(xShift)/fViewport.Width() * 360.0;
   fVRotate -= static_cast<float>(yShift)/fViewport.Height() * 180.0;
   if ( fVRotate > 0.0 ) {
      fVRotate = 0.0;
   }
   if ( fVRotate < -180.0 ) {
      fVRotate = -180.0;
   }

   fCacheDirty = kTRUE;
   return kTRUE;
}

//______________________________________________________________________________
void TGLPerspectiveCamera::Apply(const TGLBoundingBox & /*box*/, const TGLRect * pickRect)
{
   glViewport(fViewport.X(), fViewport.Y(), fViewport.Width(), fViewport.Height());

   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();

   // Load up any picking rect
   if (pickRect) {
      //TODO: Convert TGLRect so this not required
      GLint viewport[4] = { fViewport.X(), fViewport.Y(), fViewport.Width(), fViewport.Height() };
      gluPickMatrix(pickRect->X(), pickRect->Y(),
                    pickRect->Width(), pickRect->Height(),
                    viewport);
   }

   if(fViewport.Width() == 0 || fViewport.Height() == 0) {
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
   } else {
      Double_t nearClip = fDolly - (fVolumeDiag/2.0);
      Double_t farClip = nearClip + fVolumeDiag;

      if (nearClip < 3.0) {
         nearClip = 3.0;
      }
      gluPerspective(fFOV, fViewport.Aspect(), nearClip, farClip);
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
      glTranslated(0, 0, -fDolly);
      glRotated(fVRotate, 1.0, 0.0, 0.0);
      glRotated(fHRotate, 0.0, 0.0, 1.0);
      glRotated(90, 0.0, 1.0, 0.0);
      glTranslated(fTruck[0], fTruck[1], fTruck[2]);
   }

   if (fCacheDirty) {
      UpdateCache();
   }
}


