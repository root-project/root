// @(#)root/gl:$Name:  $:$Id: TGLPerspectiveCamera.cxx,v 1.9 2005/11/09 10:13:36 brun Exp $
// Author:  Richard Maunder  25/05/2005

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
#include "TError.h"

// TODO: Find somewhere else
#define PI 3.141592654

ClassImp(TGLPerspectiveCamera)

Double_t TGLPerspectiveCamera::fgFOVMin = 0.01;
Double_t TGLPerspectiveCamera::fgFOVDefault = 30;
Double_t TGLPerspectiveCamera::fgFOVMax = 120.0;

UInt_t   TGLPerspectiveCamera::fgDollyDeltaSens = 500;
UInt_t   TGLPerspectiveCamera::fgFOVDeltaSens = 500;

//______________________________________________________________________________
TGLPerspectiveCamera::TGLPerspectiveCamera() :
   fDollyMin(1.0), fDollyDefault(10.0), fDollyMax(100.0),
   fFOV(fgFOVDefault), fDolly(fDollyDefault), 
   fVRotate(0.0), fHRotate(0.0), 
   fCenter(0.0, 0.0, 0.0), fTruck(0.0, 0.0, 0.0)
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
   fDollyMin = -fDollyDefault;
   fDollyDefault *= 1.2;
   fDollyMax = fDollyDefault * 7.0;

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
Bool_t TGLPerspectiveCamera::Zoom(Int_t shift, Bool_t mod1, Bool_t mod2)
{
   if (AdjustAndClampVal(fFOV, fgFOVMin, fgFOVMax, shift, fgFOVDeltaSens, mod1, mod2)) {
      fCacheDirty = kTRUE;
      return kTRUE;
   } else {
      return kFALSE;
   }
}

//______________________________________________________________________________
Bool_t TGLPerspectiveCamera::Dolly(Int_t shift, Bool_t mod1, Bool_t mod2)
{
   if (AdjustAndClampVal(fDolly, fDollyMin, fDollyMax, shift, fgDollyDeltaSens, mod1, mod2)) {
      fCacheDirty = kTRUE;
      return kTRUE;
   } else {
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
   fTruck = fTruck + truckDelta/2.0;
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
void TGLPerspectiveCamera::Apply(const TGLBoundingBox & sceneBox, const TGLRect * pickRect)
{
   glViewport(fViewport.X(), fViewport.Y(), fViewport.Width(), fViewport.Height());

   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();

   if(fViewport.Width() == 0 || fViewport.Height() == 0) {
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
      return;
   }

   // To find decent near/far clip plane distances we construct the
   // frustum thus:
   // i) first setup perspective with arbitary near/far planes
   gluPerspective(fFOV, fViewport.Aspect(), 1.0, 1000.0);

   // ii) setup modelview
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
   glTranslated(0, 0, -fDolly);
   glRotated(fVRotate, 1.0, 0.0, 0.0);
   glRotated(fHRotate, 0.0, 0.0, 1.0);
   glRotated(90, 0.0, 1.0, 0.0);
   glTranslated(fTruck[0], fTruck[1], fTruck[2]);

   // iii) update the cached frustum planes so we can get eye point/direction
   Bool_t modifiedCache = kFALSE;
   if (fCacheDirty) {
      UpdateCache();
      modifiedCache = kTRUE;
   }

   // iv) Create a clip plane, using the eye direction as normal, passing through eye point
   TGLPlane clipPlane(EyeDirection(), EyePoint());
   fCacheDirty = modifiedCache;

   // v) find the near/far distance which just encapsulate the passed bounding box vertexes
   //    not ideal - should really find the nearest/further points on box surface
   //    which intersect frustum - however this much more complicated
   Double_t currentDist, nearClipDist=0, farClipDist=0;
   for (UInt_t i=0; i<8; i++) {
      currentDist = clipPlane.DistanceTo(sceneBox[i]);
      if (i==0) {
         nearClipDist = currentDist;
         farClipDist = nearClipDist;
      }
      if (currentDist < nearClipDist) {
         nearClipDist = currentDist;
      }
      if (currentDist > farClipDist) {
         farClipDist = currentDist;
      }
   }
   // Add 1% each way to avoid any rounding conflicts with drawn objects
   nearClipDist *= .49; //0.99; TODO Look at - avoid removing clipping + manip objs
   farClipDist *= 2.01; // 1.01
   if (farClipDist < 2.0) {
      farClipDist = 2.0;
   }
   if (nearClipDist < farClipDist/1000.0) {
      nearClipDist = farClipDist/1000.0;
   }

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
   // vi) reset the perspective using the correct near/far clips distances
   // and restore modelview mode
   gluPerspective(fFOV, fViewport.Aspect(), nearClipDist, farClipDist);
   glMatrixMode(GL_MODELVIEW);

   if (fCacheDirty) {
      UpdateCache();
   }
   // Tracing
   if (gDebug>3) {
      Info("TGLPerspectiveCamera::Apply", "FOV %f Dolly %f fVRot %f fHRot", fFOV, fDolly, fVRotate, fHRotate);
      Info("TGLPerspectiveCamera::Apply", "fTruck (%f,%f,%f)", fTruck[0], fTruck[1], fTruck[2]);
      Info("TGLPerspectiveCamera::Apply", "Near %f Far %f", nearClipDist, farClipDist);
   }
}
