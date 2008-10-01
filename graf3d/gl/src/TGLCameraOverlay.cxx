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
void TGLCameraOverlay::RenderPlaneIntersect(TGLRnrCtx& rnrCtx, TGLVertex3 &v, const TGLFont &font)
{
   // Print corss section coordinates in top right corner of screen.

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

   // get font bounding box
   Float_t bb[6];
   font.BBox(txt, bb[0], bb[1], bb[2], bb[3], bb[4], bb[5]);
   TGLRect &vp = rnrCtx.GetCamera()->RefViewport();
   Float_t off = fAxisAtt.GetAbsLabelFontSize()*0.25 ;
   Float_t x =  vp.Width()  -bb[3] -off;
   Float_t y =  vp.Height() -bb[4] -off;


   // background polygon
   TGLViewer* vw = dynamic_cast<TGLViewer*> (rnrCtx.GetViewer());
   TGLUtil::Color(vw->GetClearColor());
   Float_t x0 = (x-off)/vp.Width();
   Float_t y0 = (y-off)/vp.Height();
   TGLCapabilitySwitch poly(GL_POLYGON_OFFSET_FILL, kTRUE);
   glPolygonOffset(0.1, 1); // move polygon back
   glBegin(GL_POLYGON);
   glVertex2f(x0, y0);
   glVertex2f(1 , y0);
   glVertex2f(1 , 1);
   glVertex2f(x0, 1);
   glEnd();

   // render font
   TGLUtil::Color(fAxisAtt.GetLabelColor());
   font.RenderBitmap(txt, x, y, 0,fAxisAtt.GetTextAlign());

   // render cross
   Float_t ce = 0.15; //empty space
   Float_t w = 0.02;  //realtice size
   glBegin(GL_LINES);
   glVertex2f(0.5 +w*ce, 0.5);
   glVertex2f(0.5 +w,    0.5);

   glVertex2f(0.5 -w*ce, 0.5);
   glVertex2f(0.5 -w, 0.5);

   Float_t h = 0.02*vp.Width()/vp.Height();
   glVertex2f(0.5, 0.5 +h*ce);
   glVertex2f(0.5, 0.5 +h);

   glVertex2f(0.5, 0.5 -h*ce);
   glVertex2f(0.5, 0.5 -h);

   glEnd();

}

//______________________________________________________________________________
void TGLCameraOverlay::RenderAxis(TGLRnrCtx& rnrCtx)
{
   // Draw axis on four edges.

   Float_t l =  -rnrCtx.GetCamera()->FrustumPlane(TGLCamera::kLeft).D();
   Float_t r =   rnrCtx.GetCamera()->FrustumPlane(TGLCamera::kRight).D();
   Float_t t =   rnrCtx.GetCamera()->FrustumPlane(TGLCamera::kTop).D();
   Float_t b =  -rnrCtx.GetCamera()->FrustumPlane(TGLCamera::kBottom).D();

   // relative tick-kmark offset in x and y direction
   Float_t tms = 0.01;

   glPushMatrix();
   glScalef(1./(r-l), 1./(t-b), 1);
   // vertical
   {
      fAxisAtt.RefTMOff(0).Set(0,0,0);
      fAxisAtt.RefDir().Set(0, 1, 0);
      Float_t off = (t-b)*0.5*(1-fAxisExtend);
      fAxisAtt.SetRng(b+off, t-off);
      fAxisAtt.RefTMOff(0).X() = (t-b)*tms;

      // left
      glPushMatrix();
      glTranslatef(0, -b, 0);
      fAxisAtt.SetTextAlign(TGLFont::kRight);
      fAxisPainter.Paint(rnrCtx, fAxisAtt);
      glPopMatrix();

      // right
      glPushMatrix();
      glTranslatef(r-l, -b, 0);
      fAxisAtt.SetTextAlign(TGLFont::kLeft);
      fAxisAtt.RefTMOff(0).X() = -fAxisAtt.RefTMOff(0).X();
      fAxisPainter.Paint(rnrCtx, fAxisAtt);
      glPopMatrix();
   }

   // horizontal
   {
      fAxisAtt.RefTMOff(0).Set(0,0,0);
      fAxisAtt.RefDir().Set(1, 0, 0);
      fAxisAtt.RefTMOff(0).Y() = (t-b)*tms;

      Float_t off = (r-l)*0.5*(1-fAxisExtend);
      fAxisAtt.SetRng(l+off, r-off);
      // bottom
      glPushMatrix();
      glTranslatef(-l, 0, 0);
      fAxisAtt.SetTextAlign(TGLFont::kCenterUp);
      fAxisPainter.Paint(rnrCtx, fAxisAtt);
      glPopMatrix();
      // top
      glPushMatrix();
      glTranslatef(-l, t-b, 0);
      fAxisAtt.SetTextAlign(TGLFont::kCenterDown);
      fAxisAtt.RefTMOff(0).Y() = - fAxisAtt.RefTMOff(0).Y();
      fAxisPainter.Paint(rnrCtx, fAxisAtt);
      glPopMatrix();
   }
   glPopMatrix();
}

//______________________________________________________________________________
void TGLCameraOverlay::RenderBar(TGLRnrCtx&  rnrCtx, const TGLFont &font)
{
   // Show frustum size with fixed screen line length and printed value.

   TGLCamera &cam = rnrCtx.RefCamera();
   Float_t barsize= 0.14;

   // factors 10, 5 and 2 are allowed
   Double_t wfrust     = cam.FrustumPlane(TGLCamera::kLeft).D() + cam.FrustumPlane(TGLCamera::kRight).D();
   Double_t barsizePix  = barsize*wfrust;
   Int_t exp = (Int_t) TMath::Floor(TMath::Log10(barsizePix));
   Double_t fact = barsizePix/TMath::Power(10, exp);
   Float_t barw;
   if (fact > 5)
   {
      barw = 5*TMath::Power(10, exp);
   }
   else if (fact > 2) {
      barw = 2*TMath::Power(10, exp);
   } else {
      barw = TMath::Power(10, exp);
   }
   Double_t red = barw/wfrust;


   TGLUtil::Color(kWhite);
   const char* txt = Form("%.*f", (exp < 0) ? -exp : 0, barw);
   Float_t bb[6];
   font.BBox(txt, bb[0], bb[1], bb[2], bb[3], bb[4], bb[5]);
   TGLRect &vp = rnrCtx.GetCamera()->RefViewport();

   glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT);
   glPushMatrix();
   Float_t off = barsize*0.1;
   glTranslatef(1-barsize -off, 1 -off -bb[4]/vp.Height(), 0);
   off = barsize*0.05;
   font.RenderBitmap(txt, 0.2, 0.2, 0, fAxisAtt.GetTextAlign());
   glTranslatef(-off, -off, 0);
   glLineWidth(2.);
   Double_t mH = barsize*0.1;
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

   if ((cam.IsPerspective()  && ! fShowPerspective) ||
       (cam.IsOrthographic() && ! fShowOrthographic))
      return;

   EMode mode = cam.IsOrthographic() ? fOrthographicMode : fPerspectiveMode;

   // get intersection point with original camera
   std::pair<Bool_t, TGLVertex3> intersection;
   if (mode == kPlaneIntersect)
   {
      // get eye line
      const TGLMatrix& mx =  cam.GetCamBase() * cam.GetCamTrans();
      TGLVertex3 d   = mx.GetTranslation();
      TGLVertex3 p = d + mx.GetBaseVec(1);
      TGLLine3  line(d, p);

      // get ref plane
      const TGLPlane rp = (fUseExternalRefPlane) ? fExternalRefPlane :
         TGLPlane(cam.GetCamBase().GetBaseVec(3), TGLVertex3());

      // get intersection
      intersection = Intersection(rp, line, kTRUE);
   }

   // reset modelview and persective matrix
   glMatrixMode(GL_PROJECTION);
   glPushMatrix();
   glLoadIdentity();
   if (rnrCtx.Selection())
   {
      TGLRect rect(*rnrCtx.GetPickRectangle());
      rnrCtx.GetCamera()->WindowToViewport(rect);
      gluPickMatrix(rect.X(), rect.Y(),
                    rect.Width(), rect.Height(), cam.RefViewport().CArr());
   }
   glMatrixMode(GL_MODELVIEW);
   glPushMatrix();
   glLoadIdentity();
   glScalef(2, 2, 1); // normalised coordinates
   glTranslatef(-0.5, -0.5, 0);

   // font size
   Int_t fs = TGLFontManager::GetFontSize(cam.RefViewport().Height()*fAxisAtt.GetLabelSize());
   fAxisAtt.SetRelativeFontSize(kFALSE);
   fAxisAtt.SetAbsLabelFontSize(fs);
   TGLFont font;
   rnrCtx.RegisterFont(fs, fAxisAtt.GetLabelFontName(), TGLFont::kPixmap, font);
   TGLCapabilitySwitch lights_off(GL_LIGHTING, kFALSE);

   switch (mode)
   {
      case kPlaneIntersect:
         if (intersection.first)
            RenderPlaneIntersect(rnrCtx, intersection.second, font);
         break;
      case kBar:
         RenderBar(rnrCtx, font);
         break;
      case kAxis:
         RenderAxis(rnrCtx);
         break;
   };

   glPopMatrix();
   glMatrixMode(GL_PROJECTION);
   glPopMatrix();
   glMatrixMode(GL_MODELVIEW);
}
