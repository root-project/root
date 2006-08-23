// @(#)root/gl:$Name:  $:$Id: TGLPerspectiveCamera.cxx,v 1.16 2006/02/23 16:44:52 brun Exp $
// Author:  Richard Maunder  25/05/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLPerspectiveCamera.h"
#include "TGLUtil.h"
#include "TGLIncludes.h"

#include "TMath.h"
#include "TError.h"

// TODO: Find somewhere else
#define PI 3.141592654

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLPerspectiveCamera                                                 //
//                                                                      //
// Perspective projection camera - with characteristic foreshortening.  //
//                                                                      //
// TODO: Currently constrains YOZ plane to be floor - this is never     //
// 'tipped'. While useful we really need to extend so can:              //
// i) Pick any one of the three natural planes of the world to be floor.//
// ii) Can use a free arcball style camera with no contraint - integrate//
// TArcBall.                                                            //
//////////////////////////////////////////////////////////////////////////

ClassImp(TGLPerspectiveCamera)

Double_t TGLPerspectiveCamera::fgFOVMin = 0.01;
Double_t TGLPerspectiveCamera::fgFOVDefault = 30;
Double_t TGLPerspectiveCamera::fgFOVMax = 120.0;

UInt_t   TGLPerspectiveCamera::fgDollyDeltaSens = 500;
UInt_t   TGLPerspectiveCamera::fgFOVDeltaSens = 500;

//______________________________________________________________________________
TGLPerspectiveCamera::TGLPerspectiveCamera(const TGLVector3 & hAxis, const TGLVector3 & vAxis) :
   fHAxis(hAxis), fVAxis(vAxis),
   fDollyMin(1.0), fDollyDefault(10.0), fDollyMax(100.0),
   fFOV(fgFOVDefault), fDolly(fDollyDefault), 
   fVRotate(-90.0), fHRotate(90.0), 
   fCenter(0.0, 0.0, 0.0), fTruck(0.0, 0.0, 0.0)
{
   // Construct perspective camera
   Setup(TGLBoundingBox(TGLVertex3(-100,-100,-100), TGLVertex3(100,100,100)));
}

//______________________________________________________________________________
TGLPerspectiveCamera::~TGLPerspectiveCamera()
{
   // Destroy perspective camera
}

//______________________________________________________________________________
void TGLPerspectiveCamera::Setup(const TGLBoundingBox & box, Bool_t reset)
{
   // Setup camera limits suitible to view the world volume defined by 'box'
   // and call Reset() to initialise camera.

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

   if (reset)
      Reset();
}

//______________________________________________________________________________
void TGLPerspectiveCamera::Reset()
{
   // Reset the camera to defaults - reframe the world volume established in Setup()
   // in default state. Note: limits defined in Setup() are not adjusted.
   fFOV = fgFOVDefault;
   fHRotate = 90.0;
   fVRotate = 0.0;
   fTruck.Set(-fCenter.X(), -fCenter.Y(), -fCenter.Z());
   fDolly = fDollyDefault;
   fCacheDirty = kTRUE;
}

//______________________________________________________________________________
Bool_t TGLPerspectiveCamera::Dolly(Int_t delta, Bool_t mod1, Bool_t mod2)
{
  // Dolly the camera - 'move camera along eye line, retaining lens focal length'.
   // Arguments are:
   //
   // 'delta' - mouse viewport delta (pixels) - +ive dolly in, -ive dolly out
   // 'mod1' / 'mod2' - sensitivity modifiers - see TGLCamera::AdjustAndClampVal()
   //
   // Returns kTRUE is redraw required (camera change), kFALSE otherwise.
   if (AdjustAndClampVal(fDolly, fDollyMin, fDollyMax, delta, fgDollyDeltaSens, mod1, mod2)) {
      fCacheDirty = kTRUE;
      return kTRUE;
   } else {
      return kFALSE;
   }
}

//______________________________________________________________________________
Bool_t TGLPerspectiveCamera::Zoom(Int_t delta, Bool_t mod1, Bool_t mod2)
{
   // Zoom the camera - 'adjust lens focal length, retaining camera position'.
   // Arguments are:
   //
   // 'delta' - mouse viewport delta (pixels) - +ive zoom in, -ive zoom out
   // 'mod1' / 'mod2' - sensitivity modifiers - see TGLCamera::AdjustAndClampVal()
   //
   // Returns kTRUE is redraw required (camera change), kFALSE otherwise.
   
   // TODO: Bring all mouse handling into camera classes - would simplify interface and
   // remove these non-generic cases.
   if (AdjustAndClampVal(fFOV, fgFOVMin, fgFOVMax, delta, fgFOVDeltaSens, mod1, mod2)) {
      fCacheDirty = kTRUE;
      return kTRUE;
   } else {
      return kFALSE;
   }
}

//______________________________________________________________________________
Bool_t TGLPerspectiveCamera::Truck(Int_t x, Int_t y, Int_t xDelta, Int_t yDelta)
{
   // Truck the camera - 'move camera parallel to film plane'. The film 
   // plane is defined by the EyePoint() / EyeDirection() pair. Define motion 
   // using center point (x/y) and delta (xDelta/yDelta) - the mouse motion. 
   // For an orthographic projection this means all objects (regardless of 
   // camera distance) track the mouse motion. 
   //
   // Returns kTRUE is redraw required (camera change), kFALSE otherwise.
   //
   // Note: Trucking is often mistakenly refered to as 'pan' or 'panning'. 
   // Panning is swivelling the camera on it's own axis - the eye point.
   
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
Bool_t TGLPerspectiveCamera::Rotate(Int_t xDelta, Int_t yDelta)
{
   // Rotate the camera round view volume center established in Setup().
   // Arguments are:
   //
   // xDelta - horizontal delta (pixels)
   // YDelta - vertical delta (pixels)
   //
   // Deltas are divided by equivalent viewport dimension and scaled
   // by full rotation - i.e. translates fraction of viewport to
   // fractional rotation.   
   fHRotate += static_cast<float>(xDelta)/fViewport.Width() * 360.0;
   fVRotate -= static_cast<float>(yDelta)/fViewport.Height() * 180.0;
   if ( fVRotate > 90.0 ) {
      fVRotate = 90.0;
   }
   if ( fVRotate < -90.0 ) {
      fVRotate = -90.0;
   }

   fCacheDirty = kTRUE;
   return kTRUE;
}

//______________________________________________________________________________
void TGLPerspectiveCamera::Apply(const TGLBoundingBox & sceneBox, const TGLRect * pickRect) const
{
   // Apply the camera to the current GL context, setting the viewport, projection
   // and modelview matricies. After this verticies etc can be directly entered
   // in the world frame. This also updates the cached frustum values, enabling
   // all the projection, overlap tests etc defined in TGLCamera to be used.
   // 
   // Arguments are:
   // 'box' - view volume box - used to adjust near/far clipping
   // 'pickRect' - optional picking rect. If non-null, restrict drawing to this
   // viewport rect.
   
   // TODO: If we retained the box from Setup first argument could be dropped?
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
   glRotated(fHRotate, 0.0, 1.0, 0.0);

   // Rotate so vertical axis is just that....
   TGLVector3 zAxis = Cross(fHAxis, fVAxis);
   TGLMatrix vertMatrix(TGLVertex3(0.0, 0.0, 0.0), zAxis, &fHAxis);
   glMultMatrixd(vertMatrix.CArr());

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

      // Tracing - only show if camera changed
      if (gDebug>3) {
         Info("TGLPerspectiveCamera::Apply", "FOV %f Dolly %f fVRot %f fHRot", fFOV, fDolly, fVRotate, fHRotate);
         Info("TGLPerspectiveCamera::Apply", "fTruck (%f,%f,%f)", fTruck[0], fTruck[1], fTruck[2]);
         Info("TGLPerspectiveCamera::Apply", "Near %f Far %f", nearClipDist, farClipDist);
      }
   }
}

//______________________________________________________________________________
void TGLPerspectiveCamera::Configure(Double_t fov, Double_t dolly, Double_t center[3], 
                                     Double_t hRotate, Double_t vRotate)
{
   // Configure the camera state
   fFOV = fov;
   fDolly = dolly;
   fCenter.Set(center[0], center[1], center[2]);
   fHRotate = hRotate;
   fVRotate = vRotate;

   // Don't generally constrain external configuration
   // However exceeding the vRotate limits or silly FOV values will 
   // cause very weird behaviour or projections so fix these
   if (fVRotate > 90.0) {
      fVRotate = 90.0;
   }
   if (fVRotate < -90.0) {
      fVRotate = -90.0;
   }
   if (fFOV > 170.0) {
      fFOV = 170.0;
   } else if (fFOV < 0.1) {
      fFOV = 0.1;
   }
   fCacheDirty = kTRUE;
}
