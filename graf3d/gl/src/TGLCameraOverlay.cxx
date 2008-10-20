// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

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
#include "TGLFontManager.h"
#include "TMath.h"

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

   fAxisPainter(),
   fAxisAtt(),
   fAxisExtend(0.8),

   fExternalRefPlane(),
   fUseExternalRefPlane(kFALSE)
{
   // Constructor.

   fAxisAtt.SetAxisColor(kWhite);
   fAxisAtt.SetLabelSize(0.02);
   fAxisAtt.SetNdivisions(810);
}

//______________________________________________________________________________
void TGLCameraOverlay::RenderPlaneIntersect(TGLRnrCtx& rnrCtx, const TGLFont &font)
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
      // get print format
      Float_t m = TMath::Sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2] );
      fAxisAtt.SetRng(-m, m);
      fAxisPainter.SetAxisAtt(&fAxisAtt);
      fAxisPainter.SetTextFormat(m);
      char l0[100];
      char l1[100];
      char l2[100];
      fAxisPainter.FormAxisValue(v[0], l0);
      fAxisPainter.FormAxisValue(v[1], l1);
      fAxisPainter.FormAxisValue(v[2], l2);
      const char* txt = Form("(%s, %s, %s)", l0, l1, l2);

      TGLUtil::Color(fAxisAtt.GetLabelColor());
      glMatrixMode(GL_PROJECTION);
      glPushMatrix();
      glLoadIdentity();

      glMatrixMode(GL_MODELVIEW);
      glPushMatrix();
      glLoadIdentity();

      Float_t bb[6];
      font.BBox(txt, bb[0], bb[1], bb[2], bb[3], bb[4], bb[5]);
      TGLRect &vp = rnrCtx.GetCamera()->RefViewport();
      Float_t off = 1.5*bb[4];
      off /= vp.Height() ;
      font.RenderBitmap(txt, 1 -off, 1-off, 0,TGLFont::kRight);

      // render cross
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

   TGLCamera &cam = rnrCtx.RefCamera();
   Int_t minPix = 5; // minimum tick-mark size in pixels
   Float_t relTM = 0.015;  // tick-mark size relative to axis range
   Float_t tms;
   TGLVertex3 worldRef;

   // vertical
   fAxisAtt.RefDir().Set(0, 1, 0);
   Float_t off = (fFrustum[3]-fFrustum[1])*0.5*(1-fAxisExtend);
   fAxisAtt.SetRng(fFrustum[1]+off, fFrustum[3]-off);
   tms = (fFrustum[2]-fFrustum[0])*relTM;
   TGLVertex3 hOff = cam.ViewportDeltaToWorld(worldRef, minPix, 0);
   if (tms > hOff.X()) tms = hOff.X();
   // left
   glPushMatrix();
   glTranslated(fFrustum[0], 0, 0);
   fAxisAtt.RefTMOff(0).Set(tms, 0, 0);
   fAxisAtt.SetTextAlign(TGLFont::kLeft);
   fAxisPainter.Paint(rnrCtx, fAxisAtt);
   glPopMatrix();
   // right
   glPushMatrix();
   glTranslatef(fFrustum[2], 0, 0);
   fAxisAtt.SetTextAlign(TGLFont::kRight);
   fAxisAtt.RefTMOff(0).Set(-tms, 0, 0);
   fAxisPainter.Paint(rnrCtx, fAxisAtt);
   glPopMatrix();

   // horizontal
   fAxisAtt.RefDir().Set(1, 0, 0);
   off = (fFrustum[2]-fFrustum[0])*0.5*(1-fAxisExtend);
   fAxisAtt.SetRng(fFrustum[0]+off, fFrustum[2]-off);
   tms = (fFrustum[3]-fFrustum[1])*relTM;
   TGLVertex3 vOff = cam.ViewportDeltaToWorld(worldRef,  0, minPix);
   if (tms > vOff.Y()) tms = vOff.Y();
   // bottom
   glPushMatrix();
   glTranslatef(0, fFrustum[1], 0);
   fAxisAtt.SetTextAlign(TGLFont::kCenterDown);
   fAxisAtt.RefTMOff(0).Set( 0, tms,  0);
   fAxisPainter.Paint(rnrCtx, fAxisAtt);
   glPopMatrix();
   // top
   glPushMatrix();
   glTranslatef(0, fFrustum[3], 0);
   fAxisAtt.SetTextAlign(TGLFont::kCenterUp);
   fAxisAtt.RefTMOff(0).Set( 0, -tms,  0);
   fAxisPainter.Paint(rnrCtx, fAxisAtt);
   glPopMatrix();
}

//______________________________________________________________________________
void TGLCameraOverlay::RenderBar(TGLRnrCtx&  rnrCtx, const TGLFont &font)
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

   TGLUtil::Color(kWhite);
   const char* txt = Form("%.*f", (exp < 0) ? -exp : 0, red);
   Float_t bb[6];
   font.BBox(txt, bb[0], bb[1], bb[2], bb[3], bb[4], bb[5]);
   TGLRect &vp = rnrCtx.GetCamera()->RefViewport();
   Double_t mH = (fFrustum[3]-fFrustum[1])*bb[4]/vp.Height();
   glPushMatrix();
   glTranslatef(fFrustum[2] -barsize, fFrustum[3] - (mH*1.5), 0);
   glRasterPos2i(0,0);
   font.Render(txt);
   glPopMatrix();

   glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT);
   glLineWidth(2.);
   glPushMatrix();
   glTranslatef (fFrustum[2] - 1.1*barsize,  fFrustum[3] - 2.1*mH, 0);
   glBegin(GL_LINES);
   // horizontal static
   glVertex3d(red, 0.,0.);
   glVertex3d(barsize, 0., 0.);
   // corner bars
   glVertex3d(barsize,  mH, 0.);
   glVertex3d(barsize, -mH, 0.);
   // marker cormer bar
   TGLUtil::Color(kRed);
   glVertex3d(0.,  mH, 0.);
   glVertex3d(0., -mH, 0.);
   // marker pointer
   glVertex3d(red, 0., 0.);
   glVertex3d(red, mH, 0.);
   //marker line
   glVertex3d(0, 0.,0.);
   glVertex3d(red, 0., 0.);
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

   const GLdouble *pm = rnrCtx.RefCamera().RefLastNoPickProjM().CArr();
   GLdouble mm[16];
   GLint    vp[4];
   glGetDoublev(GL_MODELVIEW_MATRIX,  mm);
   glGetIntegerv(GL_VIEWPORT, vp);
   GLdouble l, r, t, b, z;
   gluUnProject(vp[0]+0.5, vp[1]+0.5, 0,  mm, pm, vp, &l, &b, &z);
   gluUnProject(vp[0]+vp[2]-0.5, vp[1]+vp[3]-0.5, 0,  mm, pm, vp, &r, &t, &z);
   fFrustum[0]=l;
   fFrustum[1]=b;
   fFrustum[2]=r;
   fFrustum[3]=t;

   // font size
   Int_t fs = TGLFontManager::GetFontSize(cam.RefViewport().Height()*fAxisAtt.GetLabelSize());
   fAxisAtt.SetRelativeFontSize(kFALSE);
   fAxisAtt.SetAbsLabelFontSize(fs);
   TGLFont font;
   rnrCtx.RegisterFont(fs, fAxisAtt.GetLabelFontName(), TGLFont::kPixmap, font);
   TGLCapabilitySwitch lights_off(GL_LIGHTING, kFALSE);

   if (cam.IsOrthographic())
   {
      if (fOrthographicMode == kBar)
         RenderBar(rnrCtx, font);
      else
         RenderAxis(rnrCtx);
   }
   else
   {
      RenderPlaneIntersect(rnrCtx, font);
   }
}
