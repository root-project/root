// @(#)root/gl:$Id$
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

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLPerspectiveCamera                                                 //
//                                                                      //
// Perspective projection camera - with characteristic foreshortening.  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TGLPerspectiveCamera)

Double_t TGLPerspectiveCamera::fgFOVMin = 0.01;
Double_t TGLPerspectiveCamera::fgFOVDefault = 30;
Double_t TGLPerspectiveCamera::fgFOVMax = 120.0;
UInt_t   TGLPerspectiveCamera::fgFOVDeltaSens = 500;

//______________________________________________________________________________
TGLPerspectiveCamera::TGLPerspectiveCamera() :
   TGLCamera(TGLVector3(-1.0, 0.0, 0.0), TGLVector3(0.0, 1.0, 0.0)),
   fFOV(fgFOVDefault)
{
   // Construct default XOZ perspective camera
   Setup(TGLBoundingBox(TGLVertex3(-100,-100,-100), TGLVertex3(100,100,100)));
   fCamTrans.MoveLF(1, fDollyDefault);
}

//______________________________________________________________________________
TGLPerspectiveCamera::TGLPerspectiveCamera(const TGLVector3 & hAxes, const TGLVector3 & vAxes) :
   TGLCamera(hAxes, vAxes),
   fFOV(fgFOVDefault)
{
   // Construct perspective camera
   Setup(TGLBoundingBox(TGLVertex3(-100,-100,-100), TGLVertex3(100,100,100)));
   fCamTrans.MoveLF(1, fDollyDefault);
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

   if (fExternalCenter == kFALSE)
   {
      if (fFixDefCenter)
      {
         SetCenterVec(fFDCenter.X(), fFDCenter.Y(), fFDCenter.Z());
      }
      else
      {
         TGLVertex3 center = box.Center();
         SetCenterVec(center.X(), center.Y(), center.Z());
      }
   }

   // At default FOV, the dolly should be set so as to encapsulate the scene.
   TGLVector3 extents = box.Extents();
   Int_t sortInd[3];
   TMath::Sort(3, extents.CArr(), sortInd);
   Double_t size = TMath::Hypot(extents[sortInd[0]], extents[sortInd[1]]);
   Double_t fov  = TMath::Min(fgFOVDefault, fgFOVDefault*fViewport.Aspect());

   fDollyDefault  = size / (2.0*TMath::Tan(fov*TMath::Pi()/360));
   fDollyDistance = 0.002 * fDollyDefault;

   if (reset)
   {
      Reset();
   }
}

//______________________________________________________________________________
void TGLPerspectiveCamera::Reset()
{
   // Reset the camera to defaults - reframe the world volume established in Setup()
   // in default state. Note: limits defined in Setup() are not adjusted.

   fFOV = fgFOVDefault;

   fCamTrans.SetIdentity();
   fCamTrans.MoveLF(1, fDollyDefault);

   IncTimeStamp();
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
      IncTimeStamp();
      return kTRUE;
   } else {
      return kFALSE;
   }
}

//______________________________________________________________________________
Bool_t TGLPerspectiveCamera::Truck(Int_t xDelta, Int_t yDelta, Bool_t mod1, Bool_t mod2)
{
   // Truck the camera - 'move camera parallel to film plane'.
   // Returns kTRUE is redraw required (camera change), kFALSE otherwise.

   Double_t lenMidClip = 0.5 * (fFarClip + fNearClip) * TMath::Tan(0.5*fFOV*TMath::DegToRad());

   Double_t xstep = xDelta * lenMidClip / fViewport.Height();
   Double_t ystep = yDelta * lenMidClip / fViewport.Height();

   xstep = AdjustDelta(xstep, 1.0, mod1, mod2);
   ystep = AdjustDelta(ystep, 1.0, mod1, mod2);

   return Truck(-xstep, -ystep);
}

//______________________________________________________________________________
void TGLPerspectiveCamera::Apply(const TGLBoundingBox & sceneBox,
                                 const TGLRect        * pickRect) const
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

   // MT This whole thing is convoluted. We can calculate camera postion
   // and look-at direction without calling unproject and seeking clipping
   // plane intersection.
   // Besides, this would give as a proper control over camera transforamtion
   // matrix.
   //
   // Much better since Oct 2007, the clipping planes stuff still
   // needs to be cleaned.

   glViewport(fViewport.X(), fViewport.Y(), fViewport.Width(), fViewport.Height());

   if(fViewport.Width() == 0 || fViewport.Height() == 0)
   {
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
      return;
   }

   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();

   // To find decent near/far clip plane distances we construct the
   // frustum thus:
   // i) first setup perspective with arbitary near/far planes
   gluPerspective(fFOV, fViewport.Aspect(), 1.0, 1000.0);
   //   printf("FIRST STEP FOV %f, aspect %f, nearClip %f, farClip %f \n", fFOV, fViewport.Aspect(), 1., 1000.);

   // ii) setup modelview
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
   TGLMatrix  mx     = fCamBase*fCamTrans;
   TGLVector3 pos    = mx.GetTranslation();
   TGLVector3 fwd    = mx.GetBaseVec(1);
   TGLVector3 center = pos - fwd;
   TGLVector3 up     = mx.GetBaseVec(3);

   gluLookAt(pos[0],    pos[1],    pos[2],
             center[0], center[1], center[2],
             up[0],     up[1],     up[2]);

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
   Double_t currentDist;
   for (UInt_t i=0; i<8; i++) {
      currentDist = clipPlane.DistanceTo(sceneBox[i]);
      if (i==0)
      {
         fNearClip = currentDist;
         fFarClip  = fNearClip;
      }
      if (currentDist < fNearClip)
         fNearClip = currentDist;
      if (currentDist > fFarClip)
         fFarClip = currentDist;
   }
   // Add 1% each way to avoid any rounding conflicts with drawn objects
   fNearClip *= 0.49; // 0.99; TODO Look at - avoid removing clipping + manip objs
   fFarClip  *= 2.01; // 1.01;
   if (fFarClip < 2.0)
      fFarClip = 2.0;
   if (fNearClip < fFarClip/1000.0)
      fNearClip = fFarClip/1000.0;

   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();

   // vi) Load up any picking rect and reset the perspective using the
   // correct near/far clips distances
   if (pickRect)
   {
      TGLRect rect(*pickRect);
      WindowToViewport(rect);
      gluPickMatrix(rect.X(), rect.Y(), rect.Width(), rect.Height(),
                    (Int_t*) fViewport.CArr());
      gluPerspective(fFOV, fViewport.Aspect(), fNearClip, fFarClip);
   }
   else
   {
      gluPerspective(fFOV, fViewport.Aspect(), fNearClip, fFarClip);
      glGetDoublev(GL_PROJECTION_MATRIX, fLastNoPickProjM.Arr());
   }

   glMatrixMode(GL_MODELVIEW);

   if (fCacheDirty) UpdateCache();
}

//______________________________________________________________________________
void TGLPerspectiveCamera::Configure(Double_t fov, Double_t dolly, Double_t center[3],
                                     Double_t hRotate, Double_t vRotate)
{
   // Configure the camera state.
   //   fov     - set directly field-of-view in degrees (default = 30);
   //   dolly   - additional move along the camera forward direction;
   //   center  - new camera center (can be 0 for no change);
   //   hRotate - additional "up/down" rotation in radians;
   //   vRotate - additional "left/right" rotation in radians.

   fFOV = fov;

   // Don't generally constrain external configuration
   // However exceeding the vRotate limits or silly FOV values will
   // cause very weird behaviour or projections so fix these

   if (fFOV > 170.0) {
      fFOV = 170.0;
   } else if (fFOV < 0.1) {
      fFOV = 0.1;
   }

   if (center)
      SetCenterVec(center[0], center[1], center[2]);

   fCamTrans.MoveLF(1, dolly);
   RotateRad(hRotate, vRotate);

   IncTimeStamp();
}
