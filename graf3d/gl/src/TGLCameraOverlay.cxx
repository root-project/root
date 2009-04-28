// @(#)root/eve:$Id$
// Author: Alja Mrak-Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLIncludes.h"
#include "TGLCameraOverlay.h"
#include "TGLViewer.h"
#include "TGLCamera.h"
#include "TGLSelectRecord.h"
#include "TGLUtil.h"
#include "TGLRnrCtx.h"
#include "TGLAxisPainter.h"

#include "TMath.h"
#include "TAxis.h"

//______________________________________________________________________________
// A GL overlay element which displays camera furstum.
//

ClassImp(TGLCameraOverlay);

//______________________________________________________________________________
TGLCameraOverlay::TGLCameraOverlay(Bool_t showOrtho, Bool_t showPersp) :
   TGLOverlayElement(),

   fShowOrthographic(showOrtho),
   fShowPerspective(showPersp),

   fOrthographicMode(kAxis),
   fPerspectiveMode(kPlaneIntersect),

   fAxisPainter(0),
   fAxis(0),
   fAxisExtend(0.9),

   fExternalRefPlane(),
   fUseExternalRefPlane(kFALSE)
{
   // Constructor.

   fAxis = new TAxis();
   fAxis->SetNdivisions(710);
   fAxis->SetLabelSize(0.018);
   fAxis->SetLabelOffset(0.01);
   fAxis->SetAxisColor(kGray+1);
   fAxis->SetLabelColor(kGray+1);

   fAxisPainter = new TGLAxisPainter();
   fAxisPainter->SetFontMode(TGLFont::kBitmap);
}

//______________________________________________________________________________
TGLCameraOverlay::~TGLCameraOverlay()
{
   // Destructor.

   delete  fAxisPainter;
   delete  fAxis;
}

//______________________________________________________________________________
TAttAxis* TGLCameraOverlay::GetAttAxis()
{
   return dynamic_cast<TAttAxis*>(fAxis);
}

//______________________________________________________________________________
void TGLCameraOverlay::RenderPlaneIntersect(TGLRnrCtx& rnrCtx)
{
   // Print corss section coordinates in top right corner of screen.
   TGLCamera &cam = rnrCtx.RefCamera();
   // get eye line
   const TGLMatrix& mx =  cam.GetCamBase() * cam.GetCamTrans();
   TGLVertex3 d   = mx.GetTranslation();
   TGLVertex3 p = d + mx.GetBaseVec(1);
   TGLLine3  line(d, p);
   // get ref plane
   const TGLPlane rp = (fUseExternalRefPlane) ? fExternalRefPlane :
      TGLPlane(cam.GetCamBase().GetBaseVec(3), TGLVertex3());
   // get intersection
   std::pair<Bool_t, TGLVertex3> intersection;
   intersection = Intersection(rp, line, kTRUE);

   if (intersection.first)
   {
      TGLVertex3 v = intersection.second;

      glMatrixMode(GL_PROJECTION);
      glPushMatrix();
      glLoadIdentity();

      glMatrixMode(GL_MODELVIEW);
      glPushMatrix();
      glLoadIdentity();

      TGLRect &vp = rnrCtx.GetCamera()->RefViewport();
      TGLFont font;
      Int_t fs = TGLFontManager::GetFontSize((vp.Width()+vp.Height())*0.01);
      rnrCtx.RegisterFont(fs, "arial", TGLFont::kPixmap, font);
      Float_t bb[6];
      const char* txt = Form("(%f, %f, %f)", v[0], v[1], v[2]);
      font.BBox(txt, bb[0], bb[1], bb[2], bb[3], bb[4], bb[5]);
      Float_t off = 1.5*bb[4];
      off /= vp.Height() ;
      TGLUtil::Color(kGray);
      font.RenderBitmap(txt, 1 -off, 1-off, 0,TGLFont::kRight);

      // render cross
      TGLUtil::Color(kRed);
      Float_t w = 0.02;  // cross size
      Float_t ce = 0.15; // empty space
      glBegin(GL_LINES);
      glVertex2f(0 +w*ce, 0);
      glVertex2f(0 +w,    0);

      glVertex2f(0 -w*ce, 0);
      glVertex2f(0 -w, 0);

      Float_t h = w*vp.Width()/vp.Height();
      glVertex2f(0, 0 +h*ce);
      glVertex2f(0, 0 +h);

      glVertex2f(0, 0 -h*ce);
      glVertex2f(0, 0 -h);
      glEnd();

      glPopMatrix();
      glMatrixMode(GL_PROJECTION);
      glPopMatrix();
      glMatrixMode(GL_MODELVIEW);
   }
}

//______________________________________________________________________________
void TGLCameraOverlay::RenderAxis(TGLRnrCtx& rnrCtx)
{
   // Draw axis on four edges.

   // All four axis has to have same font.
   // Size of font calculated relative to viewport diagonal
   fAxisPainter->SetAttAxis(fAxis);
   GLint   vp[4]; glGetIntegerv(GL_VIEWPORT, vp);
   Float_t rl = 0.5 *((vp[2]-vp[0]) + (vp[3]-vp[1]));
   Int_t fsize = (Int_t)(fAxis->GetLabelSize()*rl);
   Float_t tlY = 0.015*rl/(vp[2]-vp[0]);
   Float_t tlX = 0.015*rl/(vp[3]-vp[1]);


   TGLVector3 xdir = rnrCtx.RefCamera().GetCamBase().GetBaseVec(2); // left
   TGLVector3 ydir = rnrCtx.RefCamera().GetCamBase().GetBaseVec(3); // up
   xdir.Normalise();
   ydir.Normalise();
   Float_t tms = 0;
   // horizontal X
   //
   {
      fAxisPainter->SetLabelPixelFontSize(fsize);
      fAxis->SetTickLength(tlX);
      fAxisPainter->RefDir() = xdir;
      Float_t axisXOff = (fFrustum[2] - fFrustum[0]) * (1 - fAxisExtend);
      fAxis->SetLimits(fFrustum[0] + axisXOff, fFrustum[2] - axisXOff);
      fAxis->SetRangeUser(fFrustum[0] + axisXOff, fFrustum[2] - axisXOff);
      tms = fFrustum[3] - fFrustum[1];
      fAxisPainter->RefTMOff(0) = ydir*tms;

      // bottom
      glPushMatrix();
      glTranslated(ydir.X()*fFrustum[1], ydir.Y()*fFrustum[1], ydir.Z()*fFrustum[1]);
      fAxisPainter->SetLabelAlign(TGLFont::kCenterDown);
      fAxisPainter->PaintAxis(rnrCtx, fAxis);
      glPopMatrix();

      // top
      glPushMatrix();
      glTranslated(ydir.X()*fFrustum[3], ydir.Y()*fFrustum[3], ydir.Z()*fFrustum[3]);
      fAxisPainter->SetLabelAlign(TGLFont::kCenterUp);
      fAxisPainter->RefTMOff(0).Negate();
      fAxisPainter->RnrLabels();
      fAxisPainter->RnrLines();
      glPopMatrix();
   }

   //
   // vertical Y
   {
      fAxis->SetTickLength(tlY);
      fAxisPainter->RefDir() = ydir;
      Float_t axisYOff = (fFrustum[3] - fFrustum[1]) * (1 - fAxisExtend);
      fAxis->SetLimits(fFrustum[1] + axisYOff, fFrustum[3] - axisYOff);
      tms = fFrustum[2] - fFrustum[0];
      fAxisPainter->RefTMOff(0) = xdir*tms;

      // left
      glPushMatrix();
      glTranslated(xdir.X()*fFrustum[0], xdir.Y()*fFrustum[0], xdir.Z()*fFrustum[0]);
      fAxisPainter->SetLabelAlign(TGLFont::kLeft);
      fAxisPainter->PaintAxis(rnrCtx, fAxis);
      glPopMatrix();
      // right
      glPushMatrix();
      glTranslated(xdir.X()*fFrustum[2], xdir.Y()*fFrustum[2], xdir.Z()*fFrustum[2]);
      fAxisPainter->SetLabelAlign(TGLFont::kRight);
      fAxisPainter->RefTMOff(0).Negate();
      fAxisPainter->RnrLabels();
      fAxisPainter->RnrLines();
      glPopMatrix();
   }
}

//______________________________________________________________________________
void TGLCameraOverlay::RenderBar(TGLRnrCtx&  rnrCtx)
{
   // Show frustum size with fixed screen line length and printed value.

   // factors 10, 5 and 2 are allowed
   Double_t wfrust     = fFrustum[2]-fFrustum[0];
   Float_t barsize= 0.14* wfrust;
   Int_t exp = (Int_t) TMath::Floor(TMath::Log10(barsize));
   Double_t fact = barsize/TMath::Power(10, exp);
   Double_t red;
   if (fact > 5)
   {
      red = 5*TMath::Power(10, exp);
   }
   else if (fact > 2)
   {
      red = 2*TMath::Power(10, exp);
   } else
   {
      red = TMath::Power(10, exp);
   }

   TGLVector3 v;
   TGLVector3 xdir = rnrCtx.RefCamera().GetCamBase().GetBaseVec(2); // left
   TGLVector3 ydir = rnrCtx.RefCamera().GetCamBase().GetBaseVec(3); // up
   xdir.Normalise();
   ydir.Normalise();

   TGLUtil::Color(rnrCtx.ColorSet().Foreground());

   const char* txt = Form("%.*f", (exp < 0) ? -exp : 0, red);
   Float_t bb[6];
   TGLFont font;
   rnrCtx.RegisterFont(12, "arial", TGLFont::kPixmap, font);
   font.BBox(txt, bb[0], bb[1], bb[2], bb[3], bb[4], bb[5]);
   TGLRect &vp = rnrCtx.GetCamera()->RefViewport();
   Double_t mH = (fFrustum[3]-fFrustum[1])*bb[4]/vp.Height();
   glPushMatrix();
   v = xdir*(fFrustum[2]-barsize) + ydir*(fFrustum[3] - mH*1.5);
   glTranslated(v.X(), v.Y(), v.Z());
   glRasterPos2i(0,0);
   TGLUtil::Color(kGray);
   font.Render(txt);
   glPopMatrix();

   glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT);
   glLineWidth(2.);
   glPushMatrix();
   Float_t xt = fFrustum[2] - 1.1*barsize;
   Float_t yt = fFrustum[3] - 2.1*mH;
   v = xdir*xt + ydir*yt;
   glTranslated(v.X(), v.Y(), v.Z());

   glBegin(GL_LINES);
   // horizontal static
   v = red*xdir;
   glVertex3dv(v.Arr());
   v = barsize*xdir;
   glVertex3dv(v.Arr());
   // corner bars end
   v = xdir*barsize + ydir*mH;
   glVertex3dv(v.Arr());
   v = xdir*barsize - ydir*mH;
   glVertex3dv(v.Arr());
   // corner bar start
   TGLUtil::Color(kRed);
   v = ydir*mH;
   glVertex3dv(v.Arr());
   v.Negate();
   glVertex3dv(v.Arr());
   // marker pointer
   v = red*ydir;
   glVertex3dv(v.Arr());
   v += ydir*mH;
   glVertex3dv(v.Arr());
   //marker line
   glVertex3d(0, 0., 0.);
   v = red*xdir;
   glVertex3dv(v.Arr());
   glEnd();
   glPopAttrib();
   glPopMatrix();
}

//______________________________________________________________________________
void TGLCameraOverlay::Render(TGLRnrCtx& rnrCtx)
{
   // Display coodinates info of current frustum.

   TGLCamera &cam = rnrCtx.RefCamera();
   if ( rnrCtx.Selection() || (cam.IsPerspective()  && ! fShowPerspective) ||
        (cam.IsOrthographic() && ! fShowOrthographic))
      return;

   TGLCapabilitySwitch lights_off(GL_LIGHTING, kFALSE);

   // Frustum size.
   TGLCamera &camera = rnrCtx.RefCamera();
   Float_t l = -camera.FrustumPlane(TGLCamera::kLeft).D();
   Float_t r =  camera.FrustumPlane(TGLCamera::kRight).D();
   Float_t t =  camera.FrustumPlane(TGLCamera::kTop).D();
   Float_t b = -camera.FrustumPlane(TGLCamera::kBottom).D();

   fFrustum[0]=l;
   fFrustum[1]=b;
   fFrustum[2]=r;
   fFrustum[3]=t;

   if (cam.IsOrthographic())
      (fOrthographicMode == kBar) ? RenderBar(rnrCtx) :  RenderAxis(rnrCtx);
   else
      RenderPlaneIntersect(rnrCtx);
}

