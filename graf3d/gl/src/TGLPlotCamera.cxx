// @(#)root/gl:$Id$
// Author: Timur Pocheptsov

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#include "TGLPlotCamera.h"
#include "TGLIncludes.h"
#include "TVirtualGL.h"

//______________________________________________________________________________
//
// Camera for TGLPlotPainter and sub-classes.

ClassImp(TGLPlotCamera);

//______________________________________________________________________________
TGLPlotCamera::TGLPlotCamera() :
   fZoom(1.), fShift(1.5), fCenter(),
   fVpChanged(kFALSE)
{
   //Construct camera for plot painters.
   fOrthoBox[0] = 1.;
   fOrthoBox[1] = 1.;
   fOrthoBox[2] = -100.;
   fOrthoBox[3] = 100.;
}

//______________________________________________________________________________
void TGLPlotCamera::SetViewport(const TGLRect &vp)
{
   //Setup viewport, if it was changed, plus reset arcball.
   
   if (vp.Width() != fViewport.Width() || vp.Height() != fViewport.Height() ||
       vp.X() != fViewport.X() || vp.Y() != fViewport.Y())
   {
      fVpChanged = kTRUE;
      fArcBall.SetBounds(vp.Width(), vp.Height());
      fViewport = vp;
      
   } else
      fVpChanged = kFALSE;
}

//______________________________________________________________________________
void TGLPlotCamera::SetViewVolume(const TGLVertex3* /* box */)
{
   //'box' is the TGLPlotPainter's back box's coordinates.
/*   fCenter[0] = (box[0].X() + box[1].X()) / 2;
   fCenter[1] = (box[0].Y() + box[2].Y()) / 2;
   fCenter[2] = (box[0].Z() + box[4].Z()) / 2;
   const Double_t maxDim = box[1].X() - box[0].X();
   fOrthoBox[0] = maxDim;
   fOrthoBox[1] = maxDim;
   fOrthoBox[2] = -100 * maxDim;//100?
   fOrthoBox[3] = 100 * maxDim;
   fShift = maxDim * 1.5;*/
}

//______________________________________________________________________________
void TGLPlotCamera::StartRotation(Int_t px, Int_t py)
{
   //User clicks somewhere (px, py).
   fArcBall.Click(TPoint(px, py));
}

//______________________________________________________________________________
void TGLPlotCamera::RotateCamera(Int_t px, Int_t py)
{
   //Mouse movement.
   fArcBall.Drag(TPoint(px, py));
}

//______________________________________________________________________________
void TGLPlotCamera::StartPan(Int_t px, Int_t py)
{
   //User clicks somewhere (px, py).
   fMousePos.fX = px;
   fMousePos.fY = fViewport.Height() - py;
}

//______________________________________________________________________________
void TGLPlotCamera::Pan(Int_t px, Int_t py)
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
void TGLPlotCamera::SetCamera()const
{
   //Viewport and projection.
   glViewport(fViewport.X(), fViewport.Y(), fViewport.Width(), fViewport.Height());

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
void TGLPlotCamera::Apply(Double_t phi, Double_t theta)const
{
   //Applies rotations and translations before drawing
   glTranslated(0., 0., -fShift);
   glMultMatrixd(fArcBall.GetRotMatrix());
   glRotated(theta - 90., 1., 0., 0.);
   glRotated(phi, 0., 0., 1.);
   glTranslated(-fTruck[0], -fTruck[1], -fTruck[2]);
//   glTranslated(-fCenter[0], -fCenter[1], -fCenter[2]);
}

//______________________________________________________________________________
Int_t TGLPlotCamera::GetX()const
{
   //viewport[0]
   return fViewport.X();
}

//______________________________________________________________________________
Int_t TGLPlotCamera::GetY()const
{
   //viewport[1]
   return fViewport.Y();
}

//______________________________________________________________________________
Int_t TGLPlotCamera::GetWidth()const
{
   //viewport[2]
   return Int_t(fViewport.Width());
}

//______________________________________________________________________________
Int_t TGLPlotCamera::GetHeight()const
{
   //viewport[3]
   return Int_t(fViewport.Height());
}

//______________________________________________________________________________
void TGLPlotCamera::ZoomIn()
{
   //Zoom in.
   fZoom /= 1.2;
}

//______________________________________________________________________________
void TGLPlotCamera::ZoomOut()
{
   //Zoom out.
   fZoom *= 1.2;
}
