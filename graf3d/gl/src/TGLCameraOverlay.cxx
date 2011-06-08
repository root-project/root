// @(#)root/gl:$Id$
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
#include "THLimitsFinder.h"

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
   fUseAxisColors(kFALSE),

   fExternalRefPlane(),
   fUseExternalRefPlane(kFALSE)
{
   // Constructor.

   fFrustum[0] = fFrustum[1] = fFrustum[2] = fFrustum[3] = 0;

   fAxis = new TAxis();
   fAxis->SetNdivisions(710);
   fAxis->SetLabelSize(0.018);
   fAxis->SetLabelOffset(0.01);
   fAxis->SetAxisColor(kGray+1);
   fAxis->SetLabelColor(kGray+1);

   fAxisPainter = new TGLAxisPainter();
   fAxisPainter->SetFontMode(TGLFont::kBitmap);
   fAxisPainter->SetUseAxisColors(kFALSE);
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
   // Get axis attributes.

   return dynamic_cast<TAttAxis*>(fAxis);
}

//______________________________________________________________________________
void TGLCameraOverlay::SetFrustum(TGLCamera& cam)
{
   // Set frustum values from given camera.

   TGLVector3 absRef(1., 1., 1.); // needed in case if orthographic camera is negative
   Float_t l = -cam.FrustumPlane(TGLCamera::kLeft).D()  * Dot(cam.GetCamBase().GetBaseVec(2), absRef);
   Float_t r =  cam.FrustumPlane(TGLCamera::kRight).D() * Dot(cam.GetCamBase().GetBaseVec(2), absRef);
   Float_t t =  cam.FrustumPlane(TGLCamera::kTop).D();
   Float_t b = -cam.FrustumPlane(TGLCamera::kBottom).D();

   fFrustum[0] = l;
   fFrustum[1] = b;
   fFrustum[2] = r;
   fFrustum[3] = t;
}

//______________________________________________________________________________
void TGLCameraOverlay::RenderPlaneIntersect(TGLRnrCtx& rnrCtx)
{
   // Draw cross section coordinates in top right corner of screen.

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
      Int_t fs = TMath::Nint(TMath::Sqrt(vp.Width()*vp.Width() + vp.Height()*vp.Height())*0.02);
      rnrCtx.RegisterFontNoScale(fs, "arial", TGLFont::kPixmap, font);
      const char* txt = Form("(%f, %f, %f)", v[0], v[1], v[2]);
      TGLUtil::Color(rnrCtx.ColorSet().Markup());
      font.Render(txt, 0.98, 0.98, 0, TGLFont::kRight, TGLFont::kBottom);

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
void TGLCameraOverlay::RenderAxis(TGLRnrCtx& rnrCtx, Bool_t grid)
{
   // Draw axis on four edges and a transparent grid.

   fAxisPainter->SetAttAxis(fAxis);
   fAxisPainter->SetUseAxisColors(fUseAxisColors);

   Color_t lineColor = fUseAxisColors ? fAxis->GetAxisColor() : rnrCtx.ColorSet().Markup().GetColorIndex();

   // font size calculated relative to viewport diagonal
   GLint   vp[4]; glGetIntegerv(GL_VIEWPORT, vp);
   Float_t rl = 0.5 *((vp[2]-vp[0]) + (vp[3]-vp[1]));
   Int_t fsizePx = (Int_t)(fAxis->GetLabelSize()*rl);
   // tick length
   Float_t tlY = 0.015*rl/(vp[2]-vp[0]);
   Float_t tlX = 0.015*rl/(vp[3]-vp[1]);
   // corner vectors
   Float_t minX, maxX;
   TGLVector3 xdir = rnrCtx.RefCamera().GetCamBase().GetBaseVec(2); xdir.Normalise(); // left
   if (fFrustum[2] > fFrustum[0] )
   {   
      minX =  fFrustum[0];
      maxX =  fFrustum[2];
   }
   else {
      xdir = -xdir;
      minX =  fFrustum[2];
      maxX =  fFrustum[0];
   }

   TGLVector3 ydir = rnrCtx.RefCamera().GetCamBase().GetBaseVec(3); ydir.Normalise(); // up
   TGLVector3 vy1 = ydir * fFrustum[1];
   TGLVector3 vy2 = ydir * fFrustum[3];

   TGLVector3 vx1 = xdir * minX;
   TGLVector3 vx2 = xdir * maxX;
   // range
   Double_t rngY = fFrustum[3] - fFrustum[1];
   Double_t rngX = maxX - minX;
   Double_t off =  TMath::Sqrt((rngX*rngX)+(rngY*rngY)) * 0.03;
   Double_t minY = fFrustum[1] + off;
   Double_t maxY = fFrustum[3] - off;
   minX += off;
   maxX -= off;

   // grid lines
   Char_t alpha = 80; //primary
   Char_t alpha2 = 90; //seconndary
   Int_t secSteps = fAxis->GetNdivisions() % 100;
   GLushort stipple =  0x5555; // 33333 more rare

   // horizontal X
   //
   fAxisPainter->SetLabelPixelFontSize(fsizePx);
   fAxis->SetTickLength(tlX);
   fAxisPainter->RefDir() = xdir;
   fAxis->SetLimits(minX, maxX);
   fAxisPainter->RefTMOff(0) = ydir*rngY;

   // bottom
   glPushMatrix();
   glTranslated(vy1.X(), vy1.Y(), vy1.Z());
   fAxisPainter->SetLabelAlign(TGLFont::kCenterH, TGLFont::kTop);
   fAxisPainter->PaintAxis(rnrCtx, fAxis);
   glPopMatrix();

   // top
   glPushMatrix();
   glTranslated(vy2.X(), vy2.Y(), vy2.Z());
   fAxisPainter->SetLabelAlign(TGLFont::kCenterH, TGLFont::kBottom);
   fAxisPainter->RefTMOff(0).Negate();
   fAxisPainter->RnrLabels();
   fAxisPainter->RnrLines();
   glPopMatrix();

   TGLUtil::LineWidth(1);
   if (grid)
   {
      TGLAxisPainter::LabVec_t& labs = fAxisPainter->RefLabVec();
      TGLVector3 tmp;
      // draw label vertical lines
      TGLUtil::ColorTransparency(lineColor, alpha);
      glBegin(GL_LINES);
      for ( TGLAxisPainter::LabVec_t::iterator i = labs.begin(); i != labs.end(); i++)
      {
         tmp = vy1 + xdir * (i->first);
         glVertex3dv(tmp.Arr());
         tmp = vy2 + xdir * (i->first);
         glVertex3dv(tmp.Arr());
      }
      glEnd();

      // secondary tick mark lines
      if (labs.size() > 1)
      {
         TGLUtil::ColorTransparency(lineColor, alpha2);
         glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT);
         glEnable(GL_LINE_STIPPLE);
         glLineStipple(1, stipple);

         glBegin(GL_LINES);
         Int_t    ondiv = 0;
         Double_t omin = 0, omax = 0, bw1 = 0;
         THLimitsFinder::Optimize(labs[0].second, labs[1].second, secSteps, omin, omax, ondiv, bw1);
         Double_t val = labs[0].second;
         while (val < fFrustum[2])
         {
            for (Int_t k=0; k<ondiv; k++)
            {
               val += bw1;
               tmp = vy1 + xdir * val;
               glVertex3dv(tmp.Arr());
               tmp = vy2 + xdir * val;
               glVertex3dv(tmp.Arr());
            }
         }
         val = labs[0].second - bw1;
         while(val > fFrustum[0])
         {
            tmp = vy1 + xdir * val;
            glVertex3dv(tmp.Arr());
            tmp = vy2 + xdir * val;
            glVertex3dv(tmp.Arr());
            val -= bw1;
         }
         glEnd();
         glPopAttrib();
      }
   } // draw grid

   //
   // vertical Y axis
   //

   fAxis->SetTickLength(tlY);
   fAxisPainter->RefDir() = ydir;
   fAxis->SetLimits(minY, maxY);
   fAxisPainter->RefTMOff(0) = xdir*rngX;
   // left
   glPushMatrix();
   glTranslated(vx1.X(), vx1.Y(), vx1.Z());
   fAxisPainter->SetLabelAlign(TGLFont::kLeft, TGLFont::kCenterV);
   fAxisPainter->PaintAxis(rnrCtx, fAxis);
   glPopMatrix();
   // right
   glPushMatrix();
   glTranslated(vx2.X(), vx2.Y(), vx2.Z());
   fAxisPainter->SetLabelAlign(TGLFont::kRight, TGLFont::kCenterV);
   fAxisPainter->RefTMOff(0).Negate();
   fAxisPainter->RnrLabels();
   fAxisPainter->RnrLines();
   glPopMatrix();

   if (grid)
   {
      TGLAxisPainter::LabVec_t& labs = fAxisPainter->RefLabVec();
      TGLVector3 tmp;
      // draw label horizontal lines
      TGLUtil::ColorTransparency(lineColor, alpha);
      glBegin(GL_LINES);
      for ( TGLAxisPainter::LabVec_t::iterator i = labs.begin(); i != labs.end(); i++)
      {
         tmp = vx1 + ydir *(i->first);
         glVertex3dv(tmp.Arr());
         tmp = vx2 + ydir *(i->first);
         glVertex3dv(tmp.Arr());
      }
      glEnd();

      // secondary tick mark lines
      if (labs.size() > 1)
      {
         TGLUtil::ColorTransparency(lineColor, alpha2);
         glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT);
         glEnable(GL_LINE_STIPPLE);
         glLineStipple(1, stipple);

         glBegin(GL_LINES);
         Int_t    ondiv;
         Double_t omin = 0, omax = 0, bw1 = 0;
         Double_t val  = 0;
         THLimitsFinder::Optimize(labs[0].second, labs[1].second, secSteps, omin, omax, ondiv, bw1);
         val = labs[0].second;
         while(val < fFrustum[3])
         {
            for(Int_t k=0; k<ondiv; k++)
            {
               val += bw1;
               tmp = vx1 + ydir *val;
               glVertex3dv(tmp.Arr());
               tmp = vx2 + ydir * val;
               glVertex3dv(tmp.Arr());
            }
         }

         val = labs[0].second - bw1;
         while(val > fFrustum[1])
         {
            tmp = vx1 + ydir *val;
            glVertex3dv(tmp.Arr());
            tmp = vx2 + ydir * val;
            glVertex3dv(tmp.Arr());
            val -= bw1;
         }
         glEnd();
         glPopAttrib();
      }
   } // draw grid
}

//______________________________________________________________________________
void TGLCameraOverlay::RenderBar(TGLRnrCtx&  rnrCtx)
{
   // Show frustum size with fixed screen line length and printed value.

   // factors 10, 5 and 2 are allowed
   Double_t wfrust     = TMath::Abs(fFrustum[2]-fFrustum[0]);
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
   font.Render(txt);
   glPopMatrix();

   glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT);
   TGLUtil::LineWidth(2.);
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

   if (rnrCtx.Selection() ||
       (cam.IsPerspective()  && ! fShowPerspective) ||
       (cam.IsOrthographic() && ! fShowOrthographic))
   {
      return;
   }

   glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT);
   glEnable(GL_BLEND);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

   TGLUtil::Color(rnrCtx.ColorSet().Markup());
   TGLCapabilitySwitch lights_off(GL_LIGHTING, kFALSE);
   Float_t old_depth_range[2];
   glGetFloatv(GL_DEPTH_RANGE, old_depth_range);

   SetFrustum(cam);

   if (cam.IsOrthographic())
   {
      switch (fOrthographicMode)
      {
         case kBar:
            glDepthRange(0, 0.1);
            RenderBar(rnrCtx);
            break;
         case kAxis:
            glDepthRange(0, 0.1);
            RenderAxis(rnrCtx, kFALSE);
            break;
         case kGridFront:
            glDepthRange(0, 0.1);
            RenderAxis(rnrCtx, kTRUE);
            break;
         case kGridBack:
            glDepthRange(1, 0.9);
            RenderAxis(rnrCtx, kTRUE);
            break;
         default:
            break;
      };
   }
   else
   {
      RenderPlaneIntersect(rnrCtx);
   }

   glDepthRange(old_depth_range[0], old_depth_range[1]);
   glPopAttrib();
}
