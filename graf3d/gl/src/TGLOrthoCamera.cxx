// @(#)root/gl:$Id$
// Author:  Richard Maunder  25/05/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TVirtualGL.h"
#include "TMath.h"

#include "TGLOrthoCamera.h"
#include "TGLIncludes.h"
#include "TGLUtil.h"


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLOrthoCamera                                                       //
//                                                                      //
// Orthographic projection camera. Currently limited to three types     //
// defined at construction time - kXOY, kXOZ, kZOY - where this refers  //
// to the viewport plane axis - e.g. kXOY has X axis horizontal, Y      //
// vertical - i.e. looking down Z axis with Y vertical.                 //
//
// The plane types restriction could easily be removed to supported     //
// arbitary ortho projections along any axis/orientation with free      //
// rotations about them.                                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TGLOrthoCamera)

UInt_t   TGLOrthoCamera::fgZoomDeltaSens = 500;

//______________________________________________________________________________
TGLOrthoCamera::TGLOrthoCamera(EType type, const TGLVector3 & hAxis, const TGLVector3 & vAxis) :
   TGLCamera(hAxis, vAxis),
   fType(type),
   fEnableRotate(kFALSE), fDollyToZoom(kTRUE),
   fZoomMin(0.001), fZoomDefault(0.78), fZoomMax(1000.0),
   fVolume(TGLVertex3(-100.0, -100.0, -100.0), TGLVertex3(100.0, 100.0, 100.0)),
   fZoom(1.0)
{
   // Construct orthographic camera.

   Setup(TGLBoundingBox(TGLVertex3(-100,-100,-100), TGLVertex3(100,100,100)));
}

//______________________________________________________________________________
TGLOrthoCamera::~TGLOrthoCamera()
{
   // Destroy orthographic camera.
}

//______________________________________________________________________________
void TGLOrthoCamera::Setup(const TGLBoundingBox & box, Bool_t reset)
{
   // Setup camera limits suitible to view the world volume defined by 'box'
   // and call Reset() to initialise camera.

   fVolume = box;

   if (fExternalCenter == kFALSE)
   {
      TGLVertex3 center = box.Center();
      SetCenterVec(center.X(), center.Y(), center.Z());
   }
   if (reset)
      Reset();
}

//______________________________________________________________________________
void TGLOrthoCamera::Reset()
{
   // Reset the camera to defaults - trucking, zooming to reframe the world volume
   // established in Setup(). Note: limits defined in Setup() are not adjusted.

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
   }

   fDollyDefault  = 1.25*0.5*TMath::Sqrt(3)*fVolume.Extents().Mag();
   fDollyDistance = 0.002 * fDollyDefault;
   fZoom   = fZoomDefault;
   fCamTrans.SetIdentity();
   fCamTrans.MoveLF(1, fDollyDefault);
   IncTimeStamp();
}

//______________________________________________________________________________
Bool_t TGLOrthoCamera::Dolly(Int_t delta, Bool_t mod1, Bool_t mod2)
{
   // Dolly the camera.
   // By default the dolly is reinterpreted to zoom, but it can be
   // changed by modifying the fDollyToZoom data-member.

   if (fDollyToZoom) {
      return Zoom(delta, mod1, mod2);
   } else {
      return TGLCamera::Dolly(delta, mod1, mod2);
   }
}

//______________________________________________________________________________
Bool_t TGLOrthoCamera::Zoom(Int_t delta, Bool_t mod1, Bool_t mod2)
{
   // Zoom the camera - 'adjust lens focal length, retaining camera position'.
   // Arguments are:
   //
   // 'delta' - mouse viewport delta (pixels) - +ive zoom in, -ive zoom out
   // 'mod1' / 'mod2' - sensitivity modifiers - see TGLCamera::AdjustAndClampVal()
   //
   // For an orthographic camera dollying and zooming are identical and both equate
   // logically to a rescaling of the viewport limits - without center shift.
   // There is no perspective foreshortening or lens 'focal length'.
   //
   // Returns kTRUE is redraw required (camera change), kFALSE otherwise.

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

//______________________________________________________________________________
void TGLOrthoCamera::SetZoomMin(Double_t z)
{
   // Set minimum zoom factor. If current zoom is less than z it is
   // set to z.

   fZoomMin = z;
   if (fZoom < fZoomMin) {
      fZoom = fZoomMin;
      IncTimeStamp();
   }
}

//______________________________________________________________________________
void TGLOrthoCamera::SetZoomMax(Double_t z)
{
   // Set maximum zoom factor. If current zoom is greater than z it
   // is set to z.

   fZoomMax = z;
   if (fZoom > fZoomMax) {
      fZoom = fZoomMax;
      IncTimeStamp();
   }
}

//______________________________________________________________________________
Bool_t TGLOrthoCamera::Truck(Int_t xDelta, Int_t yDelta, Bool_t mod1, Bool_t mod2)
{
   // Truck the camera - 'move camera parallel to film plane'.
   // Returns kTRUE is redraw required (camera change), kFALSE otherwise.

   Double_t xstep = 2.0 * xDelta / fProjM[0] / fViewport.Width();
   Double_t ystep = 2.0 * yDelta / fProjM[5] / fViewport.Height();

   xstep = AdjustDelta(xstep, 1.0, mod1, mod2);
   ystep = AdjustDelta(ystep, 1.0, mod1, mod2);

   fCamTrans.MoveLF(2, -xstep);
   fCamTrans.MoveLF(3, -ystep);

   IncTimeStamp();
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLOrthoCamera::Rotate(Int_t xDelta, Int_t yDelta, Bool_t mod1, Bool_t mod2)
{
   // Rotate the camera - 'swivel round the view volume center'.
   // Returns kTRUE is redraw required (camera change), kFALSE otherwise.

   if (fEnableRotate)
      return TGLCamera::Rotate(xDelta, yDelta, mod1, mod2);
   else
      return kFALSE;
}

//______________________________________________________________________________
void TGLOrthoCamera::Apply(const TGLBoundingBox & /*box*/,
                           const TGLRect        * pickRect) const
{
   // Apply the camera to the current GL context, setting the viewport, projection
   // and modelview matricies. After this verticies etc can be directly entered
   // in the world frame. This also updates the cached frustum values, enabling
   // all the projection, overlap tests etc defined in TGLCamera to be used.
   //
   // Arguments are:
   // 'box' - view volume box - ignored for ortho camera. Assumed to be same
   // as one passed to Setup().
   // 'pickRect' - optional picking rect. If non-null, restrict drawing to this
   // viewport rect.

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
   TGLVector3 up     = fCamBase.GetBaseVec(3);

   gluLookAt(pos[0],    pos[1],    pos[2],
             center[0], center[1], center[2],
             up[0],     up[1],     up[2]);

   if (fCacheDirty) UpdateCache();
}


//______________________________________________________________________________
void TGLOrthoCamera::Configure(Double_t zoom, Double_t dolly, Double_t center[3],
                                     Double_t hRotate, Double_t vRotate)
{
   // Configure the camera state.

   fZoom = zoom;
   SetCenterVec(center[0], center[1], center[2]);
   fCamTrans.MoveLF(1, dolly);
   RotateRad(hRotate, vRotate);

   IncTimeStamp();
}

//////////////////////////////////////////////////////////////////////////////
//
//   TGLPlotPainter
//
/////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
TGLOrthoCamera::TGLOrthoCamera() :
   fZoomMin(0.01), fZoomDefault(0.78), fZoomMax(1000.0),
   fVolume(TGLVertex3(-100.0, -100.0, -100.0), TGLVertex3(100.0, 100.0, 100.0)),
   fZoom(1.0), fShift(0.), fCenter(),
   fVpChanged(kFALSE)
{
   // Construct orthographic camera.
   fOrthoBox[0] = 1.;
   fOrthoBox[1] = 1.;
   fOrthoBox[2] = -1.;
   fOrthoBox[3] = 1.;
}

//______________________________________________________________________________
void TGLOrthoCamera::SetViewport(TGLPaintDevice *dev)
//void TGLOrthoCamera::SetViewport(Int_t context)
{
   //Setup viewport, if it was changed, plus reset arcball.
   Int_t vp[4] = {0};
//   gGLManager->ExtractViewport(context, vp);
   dev->ExtractViewport(vp);
   if (vp[2] != Int_t(fViewport.Width()) || vp[3] != Int_t(fViewport.Height()) ||
       vp[0] != fViewport.X() || vp[1] != fViewport.Y())
   {
      fVpChanged = kTRUE;
      fArcBall.SetBounds(vp[2], vp[3]);
      fViewport.Set(vp[0], vp[1], vp[2], vp[3]);
   } else
      fVpChanged = kFALSE;
}

//______________________________________________________________________________
void TGLOrthoCamera::SetViewVolume(const TGLVertex3 *box)
{
   //'box' is the TGLPlotPainter's back box's coordinates.
   fCenter[0] = (box[0].X() + box[1].X()) / 2;
   fCenter[1] = (box[0].Y() + box[2].Y()) / 2;
   fCenter[2] = (box[0].Z() + box[4].Z()) / 2;
   const Double_t maxDim = box[1].X() - box[0].X();
   fOrthoBox[0] = maxDim;
   fOrthoBox[1] = maxDim;
   fOrthoBox[2] = -100 * maxDim;//100?
   fOrthoBox[3] = 100 * maxDim;
   fShift = maxDim * 1.5;
}

//______________________________________________________________________________
void TGLOrthoCamera::StartRotation(Int_t px, Int_t py)
{
   //User clicks somewhere (px, py).
   fArcBall.Click(TPoint(px, py));
}

//______________________________________________________________________________
void TGLOrthoCamera::RotateCamera(Int_t px, Int_t py)
{
   //Mouse movement.
   fArcBall.Drag(TPoint(px, py));
}

//______________________________________________________________________________
void TGLOrthoCamera::StartPan(Int_t px, Int_t py)
{
   //User clicks somewhere (px, py).
   fMousePos.fX = px;
   fMousePos.fY = fViewport.Height() - py;
}

//______________________________________________________________________________
void TGLOrthoCamera::Pan(Int_t px, Int_t py)
{
   //Pan camera.
   py = fViewport.Height() - py;
   //Extract gl matrices.
   Double_t mv[16] = {0.};
   glGetDoublev(GL_MODELVIEW_MATRIX, mv);
   Double_t pr[16] = {0.};
   glGetDoublev(GL_PROJECTION_MATRIX, pr);
   Int_t vp[] = {0, 0, fViewport.Width(), fViewport.Height()};
   //Adjust pan vector.
   TGLVertex3 start, end;
   gluUnProject(fMousePos.fX, fMousePos.fY, 1., mv, pr, vp, &start.X(), &start.Y(), &start.Z());
   gluUnProject(px, py, 1., mv, pr, vp, &end.X(), &end.Y(), &end.Z());
   fTruck += (start - end) /= 2.;
   fMousePos.fX = px;
   fMousePos.fY = py;
}

//______________________________________________________________________________
void TGLOrthoCamera::SetCamera()const
{
   //Viewport and projection.
   glViewport(0, 0, fViewport.Width(), fViewport.Height());

   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   glOrtho(
           -fOrthoBox[0] * fZoom,
            fOrthoBox[0] * fZoom,
           -fOrthoBox[1] * fZoom,
            fOrthoBox[1] * fZoom,
            fOrthoBox[2],
            fOrthoBox[3]
          );

   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
}

//______________________________________________________________________________
void TGLOrthoCamera::Apply(Double_t phi, Double_t theta)const
{
   //Applies rotations and translations before drawing
   glTranslated(0., 0., -fShift);
   glMultMatrixd(fArcBall.GetRotMatrix());
   glRotated(theta - 90., 1., 0., 0.);
   glRotated(phi, 0., 0., 1.);
   glTranslated(-fTruck[0], -fTruck[1], -fTruck[2]);
   glTranslated(-fCenter[0], -fCenter[1], -fCenter[2]);
}

//______________________________________________________________________________
Int_t TGLOrthoCamera::GetX()const
{
   //viewport[0]
   return fViewport.X();
}

//______________________________________________________________________________
Int_t TGLOrthoCamera::GetY()const
{
   //viewport[1]
   return fViewport.Y();
}


//______________________________________________________________________________
Int_t TGLOrthoCamera::GetWidth()const
{
   //viewport[2]
   return Int_t(fViewport.Width());
}

//______________________________________________________________________________
Int_t TGLOrthoCamera::GetHeight()const
{
   //viewport[3]
   return Int_t(fViewport.Height());
}

//______________________________________________________________________________
void TGLOrthoCamera::ZoomIn()
{
   //Zoom in.
   fZoom /= 1.2;
}

//______________________________________________________________________________
void TGLOrthoCamera::ZoomOut()
{
   //Zoom out.
   fZoom *= 1.2;
}

