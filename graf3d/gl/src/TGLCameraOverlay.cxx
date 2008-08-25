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
#include "THLimitsFinder.h"


//______________________________________________________________________________
// A GL overlay element which displays camera furstum. 
//

ClassImp(TGLCameraOverlay);

//______________________________________________________________________________
TGLCameraOverlay::TGLCameraOverlay() :
   TGLOverlayElement(),

   fAxisPainter(),
   fAxisAtt(),
   fAxisExtend(0.8),

   fExternalRefPlane(),
   fUseExternalRefPlane(kFALSE),

   fShowPerspective(kTRUE),
   fShowOrthographic(kTRUE)
{
   // Constructor.

   fAxisAtt.SetAxisColor(kWhite);
   fAxisAtt.SetLabelSize(0.02);
   fAxisAtt.SetNdivisions(810);
}


//______________________________________________________________________________
void TGLCameraOverlay::RenderPerspective(TGLRnrCtx& rnrCtx, TGLVertex3 &v, const TGLFont &font)
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
   glBegin(GL_POLYGON);
   glVertex2f(x0, y0);
   glVertex2f(1 , y0);
   glVertex2f(1 , 1);
   glVertex2f(x0, 1);
   glEnd();

   // render font
   TGLUtil::Color(fAxisAtt.GetLabelColor());
   font.PreRender();
   glPushMatrix();
   glRasterPos2i(0, 0);
   glBitmap(0, 0, 0, 0, x, y, 0);
   font.Render(txt);
   glPopMatrix();
   font.PostRender();

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
void TGLCameraOverlay::RenderOrthographic(TGLRnrCtx& rnrCtx)
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
      fAxisAtt.SetTextAlign(TGLAxisAttrib::kRight);
      fAxisPainter.Paint(rnrCtx, fAxisAtt);
      glPopMatrix();

      // right
      glPushMatrix();
      glTranslatef(r-l, -b, 0);
      fAxisAtt.SetTextAlign(TGLAxisAttrib::kLeft);
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
      fAxisAtt.SetTextAlign(TGLAxisAttrib::kCenterUp);
      fAxisPainter.Paint(rnrCtx, fAxisAtt);
      glPopMatrix();
      // top
      glPushMatrix();
      glTranslatef(-l, t-b, 0);
      fAxisAtt.SetTextAlign(TGLAxisAttrib::kCenterDown);
      fAxisAtt.RefTMOff(0).Y() = - fAxisAtt.RefTMOff(0).Y();
      fAxisPainter.Paint(rnrCtx, fAxisAtt);
      glPopMatrix();
   }
   glPopMatrix();
}

//______________________________________________________________________________
void TGLCameraOverlay::Render(TGLRnrCtx& rnrCtx)
{
   // Display coodinates info of current frustum.

   TGLCapabilitySwitch lights_off(GL_LIGHTING, kFALSE);
   TGLCamera &cam = rnrCtx.RefCamera();

   // get intersection point with original camera
   std::pair<Bool_t, TGLVertex3> intersection;
   if (cam.IsPerspective() && fShowPerspective)
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

   glMatrixMode(GL_MODELVIEW);
   glPushMatrix();
   glLoadIdentity();
   glScalef(2, 2, 1); // normalised coordinates
   glTranslatef(-0.5, -0.5, 0);

   // font size
   Int_t fs = Int_t(cam.RefViewport().Height()*fAxisAtt.GetLabelSize());
   fAxisAtt.SetAbsLabelFontSize( TGLFontManager::GetFontSize(fs, 12, 36));
   TGLFont font;
   rnrCtx.RegisterFont(12, fAxisAtt.GetLabelFontName(), TGLFont::kPixmap, font);

   if (cam.IsOrthographic())
   {
      if (fShowOrthographic)
         RenderOrthographic(rnrCtx);
   }
   else if ( intersection.first && fShowPerspective)
   {
      RenderPerspective(rnrCtx, intersection.second, font);
   }
   glPopMatrix();
   glMatrixMode(GL_PROJECTION);
   glPopMatrix();
   glMatrixMode(GL_MODELVIEW);
}
