// @(#)root/gl:$Id$
// Author:  Richard Maunder  25/05/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TMath.h"

#include "TGLOrthoCamera.h"
#include "TGLIncludes.h"
#include "TGLUtil.h"


/** \class TGLOrthoCamera
\ingroup opengl
Orthographic projection camera. Currently limited to three types
defined at construction time - kXOY, kXOZ, kZOY - where this refers
to the viewport plane axis - e.g. kXOY has X axis horizontal, Y
vertical - i.e. looking down Z axis with Y vertical.

The plane types restriction could easily be removed to supported
arbitrary ortho projections along any axis/orientation with free
rotations about them.
*/

ClassImp(TGLOrthoCamera);

UInt_t   TGLOrthoCamera::fgZoomDeltaSens = 500;

////////////////////////////////////////////////////////////////////////////////
/// Construct kXOY orthographic camera.

TGLOrthoCamera::TGLOrthoCamera() :
   TGLCamera(TGLVector3( 0.0, 0.0, 1.0), TGLVector3(0.0, 1.0, 0.0)),
   fType(kXOY),
   fEnableRotate(kFALSE), fDollyToZoom(kTRUE),
   fZoomMin(0.001), fZoomDefault(0.78), fZoomMax(1000.0),
   fVolume(TGLVertex3(-100.0, -100.0, -100.0), TGLVertex3(100.0, 100.0, 100.0)),
   fZoom(1.0)
{
   Setup(TGLBoundingBox(TGLVertex3(-100,-100,-100), TGLVertex3(100,100,100)));
}

////////////////////////////////////////////////////////////////////////////////
/// Construct orthographic camera.

TGLOrthoCamera::TGLOrthoCamera(EType type, const TGLVector3 & hAxis, const TGLVector3 & vAxis) :
   TGLCamera(hAxis, vAxis),
   fType(type),
   fEnableRotate(kFALSE), fDollyToZoom(kTRUE),
   fZoomMin(0.001), fZoomDefault(0.78), fZoomMax(1000.0),
   fVolume(TGLVertex3(-100.0, -100.0, -100.0), TGLVertex3(100.0, 100.0, 100.0)),
   fZoom(1.0)
{
   Setup(TGLBoundingBox(TGLVertex3(-100,-100,-100), TGLVertex3(100,100,100)));
}

////////////////////////////////////////////////////////////////////////////////
/// Destroy orthographic camera.

TGLOrthoCamera::~TGLOrthoCamera()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Setup camera limits suitable to view the world volume defined by 'box'
/// and call Reset() to initialise camera.

void TGLOrthoCamera::Setup(const TGLBoundingBox & box, Bool_t reset)
{
   fVolume = box;

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
   if (reset)
      Reset();
}

////////////////////////////////////////////////////////////////////////////////
/// Reset the camera to defaults - trucking, zooming to reframe the world volume
/// established in Setup(). Note: limits defined in Setup() are not adjusted.

void TGLOrthoCamera::Reset()
{
   TGLVector3 e = fVolume.Extents();
   switch (fType) {
      case kXOY:
      case kXnOY:
      {
         // X -> X, Y -> Y, Z -> Z
         fDefXSize = e.X(); fDefYSize = e.Y();
         break;
      }
      case kXOZ:
      case kXnOZ:
      {
         // X -> X, Z -> Y, Y -> Z
         fDefXSize = e.X(); fDefYSize = e.Z();
         break;
      }

      case kZOY:
      case kZnOY:
      {
         // Z -> X, Y -> Y, X -> Z
         fDefXSize = e.Z(); fDefYSize = e.Y();
         break;
      }
      case kZOX:
      case kZnOX:
      {
         // Z -> X, X -> Y, Y -> Z
         fDefXSize = e.Z(); fDefYSize = e.X();
         break;
      }
   }

   fDollyDefault  = 1.25*0.5*TMath::Sqrt(3)*fVolume.Extents().Mag();
   fDollyDistance = 0.002 * fDollyDefault;
   fZoom   = fZoomDefault;
   fCamTrans.SetIdentity();
   fCamTrans.MoveLF(1, fDollyDefault);
   IncTimeStamp();
}

////////////////////////////////////////////////////////////////////////////////
/// Dolly the camera.
/// By default the dolly is reinterpreted to zoom, but it can be
/// changed by modifying the fDollyToZoom data-member.

Bool_t TGLOrthoCamera::Dolly(Int_t delta, Bool_t mod1, Bool_t mod2)
{
   if (fDollyToZoom) {
      return Zoom(delta, mod1, mod2);
   } else {
      return TGLCamera::Dolly(delta, mod1, mod2);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Zoom the camera - 'adjust lens focal length, retaining camera position'.
/// Arguments are:
///
///  - 'delta' - mouse viewport delta (pixels) - +ive zoom in, -ive zoom out
///  - 'mod1' / 'mod2' - sensitivity modifiers - see TGLCamera::AdjustAndClampVal()
///
/// For an orthographic camera dollying and zooming are identical and both equate
/// logically to a rescaling of the viewport limits - without center shift.
/// There is no perspective foreshortening or lens 'focal length'.
///
/// Returns kTRUE is redraw required (camera change), kFALSE otherwise.

Bool_t TGLOrthoCamera::Zoom(Int_t delta, Bool_t mod1, Bool_t mod2)
{
   if (AdjustAndClampVal(fZoom, fZoomMin, fZoomMax, -delta*2, fgZoomDeltaSens, mod1, mod2))
   {
      IncTimeStamp();
      return kTRUE;
   }
   else
   {
      return kFALSE;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set minimum zoom factor. If current zoom is less than z it is
/// set to z.

void TGLOrthoCamera::SetZoomMin(Double_t z)
{
   fZoomMin = z;
   if (fZoom < fZoomMin) {
      fZoom = fZoomMin;
      IncTimeStamp();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set maximum zoom factor. If current zoom is greater than z it
/// is set to z.

void TGLOrthoCamera::SetZoomMax(Double_t z)
{
   fZoomMax = z;
   if (fZoom > fZoomMax) {
      fZoom = fZoomMax;
      IncTimeStamp();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Truck the camera - 'move camera parallel to film plane'.
/// Returns kTRUE is redraw required (camera change), kFALSE otherwise.

Bool_t TGLOrthoCamera::Truck(Int_t xDelta, Int_t yDelta, Bool_t mod1, Bool_t mod2)
{
   Double_t xstep = 2.0 * xDelta / fProjM[0] / fViewport.Width();
   Double_t ystep = 2.0 * yDelta / fProjM[5] / fViewport.Height();

   xstep = AdjustDelta(xstep, 1.0, mod1, mod2);
   ystep = AdjustDelta(ystep, 1.0, mod1, mod2);

   return Truck(-xstep, -ystep);
}

////////////////////////////////////////////////////////////////////////////////
/// Rotate the camera - 'swivel round the view volume center'.
/// Returns kTRUE is redraw required (camera change), kFALSE otherwise.

Bool_t TGLOrthoCamera::Rotate(Int_t xDelta, Int_t yDelta, Bool_t mod1, Bool_t mod2)
{
   if (fEnableRotate)
      return TGLCamera::Rotate(xDelta, yDelta, mod1, mod2);
   else
      return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Apply the camera to the current GL context, setting the viewport, projection
/// and modelview matrices. After this vertices etc can be directly entered
/// in the world frame. This also updates the cached frustum values, enabling
/// all the projection, overlap tests etc defined in TGLCamera to be used.
///
/// Arguments are:
///  - 'box' - view volume box - ignored for ortho camera. Assumed to be same
///     as one passed to Setup().
///  - 'pickRect' - optional picking rect. If non-null, restrict drawing to this
///     viewport rect.

void TGLOrthoCamera::Apply(const TGLBoundingBox & /*box*/,
                           const TGLRect        * pickRect) const
{
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

   // Load up any picking rect
   if (pickRect)
   {
      TGLRect rect(*pickRect);
      WindowToViewport(rect);
      gluPickMatrix(rect.X(), rect.Y(), rect.Width(), rect.Height(),
                    (Int_t*) fViewport.CArr());
   }

   Double_t halfRangeX, halfRangeY;
   if (fDefYSize*fViewport.Width()/fDefXSize > fViewport.Height()) {
      halfRangeY = 0.5 *fDefYSize;
      halfRangeX = halfRangeY*fViewport.Width()/fViewport.Height();
   } else {
      halfRangeX = 0.5 *fDefXSize;
      halfRangeY = halfRangeX*fViewport.Height()/fViewport.Width();
   }

   halfRangeX /= fZoom;
   halfRangeY /= fZoom;

   fNearClip = 0.05*fDollyDefault;
   fFarClip  = 2.0*fDollyDefault;
   glOrtho(-halfRangeX, halfRangeX,
           -halfRangeY, halfRangeY,
            fNearClip,  fFarClip);

   if (!pickRect) glGetDoublev(GL_PROJECTION_MATRIX, fLastNoPickProjM.Arr());

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

   if (fCacheDirty) UpdateCache();
}

////////////////////////////////////////////////////////////////////////////////
/// Configure the camera state.
///  - zoom    - set directly (default = 0.78);
///  - dolly   - additional move along the camera forward direction;
///  - center  - new camera center (can be 0 for no change);
///  - hRotate - additional "up/down" rotation in radians;
///  - vRotate - additional "left/right" rotation in radians.

void TGLOrthoCamera::Configure(Double_t zoom, Double_t dolly, Double_t center[3],
                               Double_t hRotate, Double_t vRotate)
{
   fZoom = zoom;

   if (center)
      SetCenterVec(center[0], center[1], center[2]);

   fCamTrans.MoveLF(1, dolly);
   RotateRad(hRotate, vRotate);

   IncTimeStamp();
}
