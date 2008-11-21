// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveProjectionAxesGL.h"
#include "TEveProjectionAxes.h"
#include "TEveProjectionManager.h"
#include "THLimitsFinder.h"

#include "TGLIncludes.h"
#include "TGLRnrCtx.h"
#include "TGLFontManager.h"
#include "TGLCamera.h"

#include "TMath.h"

//______________________________________________________________________________
//
// OpenGL renderer class for TEveProjectionAxes.
//

ClassImp(TEveProjectionAxesGL);

//______________________________________________________________________________
TEveProjectionAxesGL::TEveProjectionAxesGL() :
   TGLObject(),
   fM(0),
   fProjection(0)
{
   // Constructor.

   fDLCache    = kFALSE; // Disable display list.
}

//______________________________________________________________________________
Bool_t TEveProjectionAxesGL::SetModel(TObject* obj, const Option_t* /*opt*/)
{
   // Set model object.
   // Virtual from TGLObject.

   if (SetModelCheckClass(obj, TEveProjectionAxes::Class())) {
      fM = dynamic_cast<TEveProjectionAxes*>(obj);
      return fM->GetManager() ? kTRUE : kFALSE;
   }
   return kFALSE;
}

//______________________________________________________________________________
void TEveProjectionAxesGL::SetBBox()
{
   // Fill the bounding-box data of the logical-shape.
   // Virtual from TGLObject.

   SetAxisAlignedBBox(((TEveProjectionAxes*)fExternalObj)->AssertBBox());
}

//______________________________________________________________________________
void TEveProjectionAxesGL::DrawScales(Bool_t horizontal, TGLFont& font, Float_t tmSize, Float_t dtw) const
{
   // Draw labels and tick-marks.

   // tick-marks

   glBegin(GL_LINES);

   // draw small tickmarks
   Float_t mh = tmSize*0.5;
   for (TMVec_t::iterator it = fTickMarks.begin(); it != fTickMarks.end(); ++it)
   {
      if (horizontal)
      {
         glVertex2f(*it, 0);  glVertex2f(*it, mh);
      }
      else
      {
         glVertex2f(0, *it); glVertex2f(mh, *it);
      }
   }

   // draw label tickmarks
   Int_t minIdx = 0;
   Float_t minVal = TMath::Abs( fLabVec[0].second);
   Int_t cnt = 0;
   for (LabVec_t::iterator it = fLabVec.begin(); it != fLabVec.end(); ++it)
   {
      if (TMath::Abs((*it).second) < minVal)
      {
         minVal = TMath::Abs((*it).second);
         minIdx = cnt;
      }

      if (horizontal)
      {
         glVertex2f((*it).first, 0);  glVertex2f((*it).first, tmSize);
      }
      else
      {
         glVertex2f(0, (*it).first); glVertex2f(tmSize, (*it).first);
      }
      cnt++;
   }
   glEnd();

   //  align labels
   TGLFont::ETextAlign_e align;
   if (horizontal)
      align = (tmSize < 0) ? TGLFont::kCenterUp : TGLFont::kCenterDown;
   else
      align = (tmSize < 0) ? TGLFont::kRight : TGLFont::kLeft;
   Float_t llx, lly, llz, urx, ury, urz;


   // get text format for current axis range and bin width
   fAxisAtt.SetRng(fLabVec.front().second, fLabVec.back().second);
   fAxisPainter.SetAxisAtt(&fAxisAtt);
   if (minIdx + 1 < (Int_t) fLabVec.size())
      fAxisPainter.SetTextFormat(fLabVec[minIdx+1].second - fLabVec[minIdx].second);
   else
      fAxisPainter.SetTextFormat(fLabVec[minIdx].second - fLabVec[minIdx-1].second);

   // move from center out to be symetric
   Int_t nl = fLabVec.size();
   char txt[100];
   Float_t off = 1.5*tmSize;


   // render zero or minimum absolute value if not exist
   fAxisPainter.FormAxisValue(fLabVec[minIdx].second, txt);
   font.BBox(txt, llx, lly, llz, urx, ury, urz);
   if (horizontal)
      font.RenderBitmap(txt, fLabVec[minIdx].first, off, 0, align);
   else
      font.RenderBitmap(txt, off, fLabVec[minIdx].first, 0, align);

   // positive values (from zero to right)
   Float_t prev = fLabVec[minIdx].first;
   for (Int_t i=minIdx+1; i<nl; ++i)
   {
      fAxisPainter.FormAxisValue(fLabVec[i].second, txt);
      font.BBox(txt, llx, lly, llz, urx, ury, urz);
      if (prev > (fLabVec[i].first - (urx-llx)*0.5*dtw))
         continue;

      if (horizontal)
         font.RenderBitmap(txt, fLabVec[i].first, off, 0, align);
      else
         font.RenderBitmap(txt, off, fLabVec[i].first, 0, align);

      prev = fLabVec[i].first + (urx-llx)*0.5*dtw;
   }

   // negative values (zero to left)
   prev = fLabVec[minIdx].first;
   for (Int_t i=minIdx-1; i>=0; --i)
   {
      fAxisPainter.FormAxisValue(fLabVec[i].second, txt);
      font.BBox(txt, llx, lly, llz, urx, ury, urz);
      if ( prev < (fLabVec[i].first + (urx-llx)*0.5*dtw ))
         continue;

      if (horizontal)
         font.RenderBitmap(txt, fLabVec[i].first, off, 0, align);
      else
         font.RenderBitmap(txt, off, fLabVec[i].first, 0, align);

      prev = fLabVec[i].first -(urx-llx)*0.5 *dtw;
   }
}

//______________________________________________________________________________
void TEveProjectionAxesGL::SplitInterval(Float_t p1, Float_t p2, Int_t ax, Int_t nLab) const
{
   // Build an array of tick-mark position-value pairs.

   fLabVec.clear();
   fTickMarks.clear();

   if (fM->GetLabMode() == TEveProjectionAxes::kValue)
   {
      SplitIntervalByVal(p1, p2, ax, nLab);
   }
   else if (fM->GetLabMode() == TEveProjectionAxes::kPosition)
   {
      SplitIntervalByPos(p1, p2, ax, nLab);
   }
}

//______________________________________________________________________________
void TEveProjectionAxesGL::SplitIntervalByPos(Float_t p1, Float_t p2, Int_t ax, Int_t nLab) const
{
   // Add tick-marks at equidistant position.

   // limits
   Int_t ndiv = fM->GetNdivisions();
   Int_t n1a = nLab;
   Int_t n2a = ndiv % 100;
   Int_t bn1, bn2;
   Double_t bw1, bw2; // bin with first second order
   Double_t bl1, bh1, bl2, bh2; // bin low, high first second order
   THLimitsFinder::Optimize(p1, p2, n1a, bl1, bh1, bn1, bw1);
   THLimitsFinder::Optimize(bl1, bl1+bw1, n2a, bl2, bh2, bn2, bw2);

   Int_t n1=TMath::CeilNint(p1/bw1);
   Int_t n2=TMath::FloorNint(p2/bw1);

   Float_t p = n1*bw1;
   Float_t pMinor = p;
   for (Int_t l=n1; l<=n2; l++)
   {
      fLabVec.push_back(Lab_t(p , fProjection->GetValForScreenPos(ax, p)));

      // tickmarks
      fTickMarks.push_back(p);
      pMinor = p-bw2;
      for (Int_t i=1; i<bn2; i++)
      {
         if (pMinor > p2)  break;
         fTickMarks.push_back(pMinor);
         pMinor += bw2;
      }
      p += bw1;
   }

   // complete
   pMinor = n1*bw1 -bw2;
   while ( pMinor > p1)
   {
      fTickMarks.push_back(pMinor);
      pMinor -=bw2;
   }
}

//______________________________________________________________________________
void TEveProjectionAxesGL::SplitIntervalByVal(Float_t p1, Float_t p2, Int_t ax, Int_t nLab) const
{
   // Add tick-marks on fixed value step.

   Float_t v1 = fProjection->GetValForScreenPos(ax, p1);
   Float_t v2 = fProjection->GetValForScreenPos(ax, p2);


   // limits
   Int_t ndiv = fM->GetNdivisions();
   Int_t n1a = nLab;
   Int_t n2a = ndiv %100;
   Int_t bn1, bn2;
   Double_t bw1, bw2; // bin with first second order
   Double_t bl1, bh1, bl2, bh2; // bin low, high first second order
   THLimitsFinder::Optimize(v1, v2, n1a, bl1, bh1, bn1, bw1);
   THLimitsFinder::Optimize(bl1, bl1+bw1, n2a, bl2, bh2, bn2, bw2);

   Float_t pMinor;
   Float_t v = bl1;
   // step
   for (Int_t l=0; l<=bn1; l++)
   {
      fLabVec.push_back(Lab_t(fProjection->GetScreenVal(ax, v) , v));

      // tickmarks
      for (Int_t k=0; k<bn2; k++)
      {
         pMinor = fProjection->GetScreenVal(ax, v+k*bw2);
         if (pMinor > p2)  break;
         fTickMarks.push_back(pMinor);
      }
      v += bw1;
   }

   // complete
   v = bl1 -bw2;
   while ( v > v1)
   {
      pMinor = fProjection->GetScreenVal(ax, v);
      if (pMinor < p1)  break;
      fTickMarks.push_back(pMinor);
      v -= bw2;
   }
}

//______________________________________________________________________________
Bool_t TEveProjectionAxesGL::GetRange(Int_t ax, Float_t frustMin, Float_t frustMax, Float_t& start, Float_t& end) const
{
   // take range from bounding box of projection manager

   Bool_t rngBBox=kTRUE;

   Float_t bf = 1.2;
   start = fM->GetManager()->GetBBox()[ax*2]   * bf;
   end   = fM->GetManager()->GetBBox()[ax*2+1] * bf;

   // compare frustum range with bbox, take smaller
   Float_t frustC = 0.5f * (frustMin + frustMax);
   Float_t frustR = 0.8f * (frustMax - frustMin);
   frustMin = frustC - 0.5f*frustR;
   frustMax = frustC + 0.5f*frustR;
   if (start < frustMin || end > frustMax)
   {
      rngBBox=kFALSE;
      start = frustMin;
      end = frustMax;
   }

   // ceheck projection  limits
   // set limit factor in case of divergence
   Float_t dLim = fProjection->GetLimit(ax, 0);
   Float_t uLim = fProjection->GetLimit(ax, 1);

   if (start < dLim) start = dLim*0.98;
   if (end   > uLim) end   = uLim*0.98;

   return rngBBox;
}

//______________________________________________________________________________
void TEveProjectionAxesGL::DirectDraw(TGLRnrCtx& rnrCtx) const
{
   // Actual rendering code.
   // Virtual from TGLLogicalShape.

   if (rnrCtx.Selection() || rnrCtx.Highlight()) return;

   fProjection = fM->GetManager()->GetProjection();

   Float_t old_depth_range[2];
   glGetFloatv(GL_DEPTH_RANGE, old_depth_range);
   glDepthRange(0, 0); // Draw on front-clipping plane.

   // frustum size
   TGLCamera &camera = rnrCtx.RefCamera();
   Float_t l = -camera.FrustumPlane(TGLCamera::kLeft).D();
   Float_t r =  camera.FrustumPlane(TGLCamera::kRight).D();
   Float_t t =  camera.FrustumPlane(TGLCamera::kTop).D();
   Float_t b = -camera.FrustumPlane(TGLCamera::kBottom).D();

   // font size relative to wiewport width
   TGLFont font;
   GLint   vp[4];
   glGetIntegerv(GL_VIEWPORT, vp);
   Int_t fs =  TGLFontManager::GetFontSize(TMath::Min(vp[2], vp[3])*fM->GetLabelSize(), 8, 36);
   rnrCtx.RegisterFont(fs, "arial", TGLFont::kPixmap, font);
   font.PreRender();

   Float_t bboxCentX = (fM->GetManager()->GetBBox()[0] + fM->GetManager()->GetBBox()[1])* 0.5;
   Float_t bboxCentY = (fM->GetManager()->GetBBox()[2] + fM->GetManager()->GetBBox()[3])* 0.5;

   Float_t startX, endX;
   Float_t startY, endY;
   Bool_t  rngBBoxX = GetRange(0, l, r, startX, endX);
   Bool_t  rngBBoxY = GetRange(1, b, t, startY, endY);
   Float_t rngX = endX - startX;
   Float_t rngY = endY - startY;

   const Float_t rtm    = 0.015; // relative tick-mark size
   const Int_t   minPix = 5;     // minimum tick-mark size in pixels

   // X-axis
   if (fM->fAxesMode == TEveProjectionAxes::kAll ||
       fM->fAxesMode == TEveProjectionAxes::kHorizontal)
   {

      Float_t dtw  = (r-l)/vp[2]; // delta to viewport
      Int_t   nLab = (rngX < rngY ) ? TMath::FloorNint(fM->GetNdivisions()/100) :
                                      TMath::CeilNint((fM->GetNdivisions()*rngX)/(rngY*100));
      SplitInterval(startX, endX, 0, nLab);

      Float_t vOff = dtw*minPix;
      Float_t tms  = (t-b)*rtm;
      if (tms < vOff) tms = vOff;

      // bottom
      glPushMatrix();
      glTranslatef(rngBBoxX ? bboxCentX : 0, b, 0);
      DrawScales(kTRUE, font, tms, dtw);
      glPopMatrix();
      // top
      glPushMatrix();
      glTranslatef(rngBBoxX ? bboxCentX : 0, t, 0);
      DrawScales(kTRUE, font, -tms, dtw);
      glPopMatrix();
   }

   // Y-axis
   if (fM->fAxesMode == TEveProjectionAxes::kAll ||
       fM->fAxesMode == TEveProjectionAxes::kVertical)
   {
      Float_t dtw = (t-b)/vp[3];// delta to viewport
      Int_t nLab = (rngY < rngX ) ? TMath::FloorNint(fM->GetNdivisions()/100) : TMath::CeilNint((fM->GetNdivisions()*rngY)/(rngX*100)) ;
      SplitInterval(startY, endY, 1, nLab);

      Float_t hOff = dtw * minPix;
      Float_t tms  = (r - l) * rtm;
      if (tms < hOff) tms = hOff;

      // left
      glPushMatrix();
      glTranslatef(l, rngBBoxY ? bboxCentY : 0, 0);
      DrawScales(kFALSE, font, tms, dtw);
      glPopMatrix();
      // right
      glPushMatrix();
      glTranslatef(r, rngBBoxY ? bboxCentY : 0, 0);
      DrawScales(kFALSE, font, -tms, dtw);
      glPopMatrix();
   }
   font.PostRender();

   // projection center and origin marker
   Float_t d = ((r-l) > (b-t)) ? (b-t) : (r-l);
   d *= 0.02f;
   if (fM->GetDrawCenter())
   {
      Float_t* c = fProjection->GetProjectedCenter();
      TGLUtil::Color3f(1., 0., 0.);
      glBegin(GL_LINES);
      glVertex3f(c[0] + d, c[1], c[2]); glVertex3f(c[0] - d, c[1], c[2]);
      glVertex3f(c[0], c[1] + d, c[2]); glVertex3f(c[0], c[1] - d, c[2]);
      glVertex3f(c[0], c[1], c[2] + d); glVertex3f(c[0], c[1], c[2] - d);
      glEnd();
   }
   if (fM->GetDrawOrigin())
   {
      TEveVector zero;
      fProjection->ProjectVector(zero);
      TGLUtil::Color3f(1., 1., 1.);
      glBegin(GL_LINES);
      glVertex3f(zero[0] + d, zero[1], zero[2]); glVertex3f(zero[0] - d, zero[1], zero[2]);
      glVertex3f(zero[0], zero[1] + d, zero[2]); glVertex3f(zero[0], zero[1] - d, zero[2]);
      glVertex3f(zero[0], zero[1], zero[2] + d); glVertex3f(zero[0], zero[1], zero[2] - d);
      glEnd();
   }

   glDepthRange(old_depth_range[0], old_depth_range[1]);
}
