// @(#)root/eve:$Id$
// Author: Alja Mrak-Tadel 2007

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

   fM = SetModelDynCast<TEveProjectionAxes>(obj);
   fAxisPainter.SetAttAxis(fM);
   return fM->GetManager() ? kTRUE : kFALSE;
}

//______________________________________________________________________________
void TEveProjectionAxesGL::SetBBox()
{
   // Fill the bounding-box data of the logical-shape.
   // Virtual from TGLObject.

   SetAxisAlignedBBox(((TEveProjectionAxes*)fExternalObj)->AssertBBox());
}

//______________________________________________________________________________
void TEveProjectionAxesGL::FilterOverlappingLabels(Int_t idx, Float_t ref) const
{
   TGLAxisPainter::LabVec_t &orig = fAxisPainter.RefLabVec();
   if (orig.size() == 0) return;

   Float_t center = fM->GetManager()->GetCenter()[idx];

   // Get index of label closest to the distortion center.
   // Needed to keep simetry around center.
   Int_t minIdx = 0;
   Int_t cnt = 0;
   Float_t currD = 0;
   Float_t minD = TMath::Abs(orig[0].first -center);
   for (TGLAxisPainter::LabVec_t::iterator it = orig.begin(); it != orig.end(); ++it)
   {
      currD = TMath::Abs((*it).first - center);
      if (minD > currD)
      {
         minD = currD;
         minIdx = cnt;
      }
      cnt++;
   }

   // Minimum allowed distance 4* font size.
   TGLAxisPainter::LabVec_t  filtered;
   filtered.push_back(orig[minIdx]);
   Int_t size = orig.size();
   Float_t minDist = 4*fM->GetLabelSize()*ref;
   Float_t pos = 0;

   // Go from center to minimum.
   if (minIdx > 0)
   {
      pos =  orig[minIdx].first;
      for (Int_t i=minIdx-1; i>=0; --i)
      {
         if (TMath::Abs(pos - orig[i].first) > minDist)
         {
            filtered.push_back(orig[i]);
            pos = orig[i].first;
         }
      }
   }

   // Go from center to maximum.
   if (minIdx < (size -1))
   {
      pos =  orig[minIdx].first;
      for (Int_t i=minIdx+1; i<size; ++i)
      {
         if (TMath::Abs(orig[i].first - pos) > minDist)
         {
            filtered.push_back(orig[i]);
            pos = orig[i].first;
         }
      }
   }

   // Set labels list and text format.
   if (filtered.size() >= 2)
   {
      if ( minIdx > 0 )
         fAxisPainter.SetTextFormat(orig.front().second, orig.back().second,  orig[minIdx].second - orig[minIdx-1].second);
      else
         fAxisPainter.SetTextFormat(orig.front().second, orig.back().second,  orig[minIdx+1].second - orig[minIdx].second);

      fAxisPainter.RefLabVec().swap(filtered);
   }
   else
   {
      fAxisPainter.SetTextFormat(orig.front().second, orig.back().second,  orig.back().second - orig.front().second);
   }
}

//______________________________________________________________________________
void TEveProjectionAxesGL::SplitInterval(Float_t p1, Float_t p2, Int_t ax) const
{
   // Build an array of tick-mark position-value pairs.

   fAxisPainter.RefLabVec().clear();
   fAxisPainter.RefTMVec().clear();

   // Get list of label position-value pairs.


   // Minimum/maximum are defined at the front/back element of list.
   fAxisPainter.RefTMVec().push_back(TGLAxisPainter::TM_t(p1, -1));

   if (fM->GetLabMode() == TEveProjectionAxes::kValue)
   {
      SplitIntervalByVal(p1, p2, ax);
   }
   else if (fM->GetLabMode() == TEveProjectionAxes::kPosition)
   {
      SplitIntervalByPos(p1, p2, ax);
   }


   FilterOverlappingLabels(0, p2 -p1);

   // Minimum/maximum are defined at the front/back element of list.
   fAxisPainter.RefTMVec().push_back(TGLAxisPainter::TM_t(p2, -1));
}

//______________________________________________________________________________
void TEveProjectionAxesGL::SplitIntervalByPos(Float_t p1, Float_t p2, Int_t ax) const
{
   // Add tick-marks at equidistant position.

   // Limits.
   Int_t n1a = TMath::FloorNint(fM->GetNdivisions() / 100);
   Int_t n2a = fM->GetNdivisions() - n1a * 100;
   Int_t bn1, bn2;
   Double_t bw1, bw2; // bin with first second order
   Double_t bl1, bh1, bl2, bh2; // bin low, high first second order
   THLimitsFinder::Optimize(p1, p2, n1a, bl1, bh1, bn1, bw1);
   THLimitsFinder::Optimize(bl1, bl1+bw1, n2a, bl2, bh2, bn2, bw2);

   Int_t n1=TMath::CeilNint(p1/bw1);
   Int_t n2=TMath::FloorNint(p2/bw1);

   TGLAxisPainter::LabVec_t &labVec = fAxisPainter.RefLabVec();
   TGLAxisPainter::TMVec_t  &tmVec =  fAxisPainter.RefTMVec();

   Float_t p = n1*bw1;
   Float_t pMinor = p;
   for (Int_t l=n1; l<=n2; l++)
   {
      // Labels.
      labVec.push_back( TGLAxisPainter::Lab_t(p , fProjection->GetValForScreenPos(ax, p)));

      // Tick-marks.
      tmVec.push_back(TGLAxisPainter::TM_t(p, 0));
      pMinor = p+bw2;
      for (Int_t i=1; i<bn2; i++)
      {
         if (pMinor > p2)  break;
         tmVec.push_back( TGLAxisPainter::TM_t(pMinor, 1));
         pMinor += bw2;
      }
      p += bw1;
   }

   // Complete second order tick-marks.
   pMinor = n1*bw1 -bw2;
   while ( pMinor > p1)
   {
      tmVec.push_back(TGLAxisPainter::TM_t(pMinor, 1));
      pMinor -=bw2;
   }
}

//______________________________________________________________________________
void TEveProjectionAxesGL::SplitIntervalByVal(Float_t p1, Float_t p2, Int_t ax) const
{
   // Add tick-marks on fixed value step.

   Float_t v1 = fProjection->GetValForScreenPos(ax, p1);
   Float_t v2 = fProjection->GetValForScreenPos(ax, p2);

   TGLAxisPainter::LabVec_t &labVec =  fAxisPainter.RefLabVec();
   TGLAxisPainter::TMVec_t  &tmVec  =  fAxisPainter.RefTMVec();

   // Limits
   Int_t n1a = TMath::FloorNint(fM->GetNdivisions() / 100);
   Int_t n2a = fM->GetNdivisions() - n1a * 100;
   Int_t bn1, bn2;
   Double_t bw1, bw2;           // bin width first / second order
   Double_t bl1, bh1, bl2, bh2; // bin low, high first / second order
   THLimitsFinder::Optimize(v1, v2, n1a, bl1, bh1, bn1, bw1);
   THLimitsFinder::Optimize(bl1, bl1+bw1, n2a, bl2, bh2, bn2, bw2);

   Float_t pFirst, pSecond; // position of first, second order of tickmarks
   Float_t v = bl1;
   // step
   for (Int_t l=0; l<=bn1; l++)
   {
      // Labels.
      pFirst = fProjection->GetScreenVal(ax, v);
      labVec.push_back(TGLAxisPainter::Lab_t(pFirst , v));
      tmVec.push_back(TGLAxisPainter::TM_t(pFirst, 0));

      // Tickmarks.
      for (Int_t k=1; k<bn2; k++)
      {
         pSecond = fProjection->GetScreenVal(ax, v+k*bw2);
         if (pSecond > p2)  break;
         tmVec.push_back(TGLAxisPainter::TM_t(pSecond, 1));
      }
      v += bw1;
   }

   // Complete second order tick-marks.
   v = bl1 -bw2;
   while ( v > v1)
   {
      pSecond = fProjection->GetScreenVal(ax, v);
      if (pSecond < p1)  break;
      tmVec.push_back(TGLAxisPainter::TM_t(pSecond, 1));
      v -= bw2;
   }
}

//______________________________________________________________________________
void TEveProjectionAxesGL::GetRange(Int_t ax, Float_t frustMin, Float_t frustMax, Float_t& min, Float_t& max) const
{
   // Get range from bounding box of projection manager


   // Compare frustum range with bbox and take larger.

   Float_t frng = (frustMax -frustMin)*0.4;
   Float_t c = 0.5*(frustMax +frustMin);
   min = c - frng;
   max = c + frng;

   // Check projection  limits.
   // Set limit factor in case of divergence.
   Float_t dLim = fProjection->GetLimit(ax, 0);
   Float_t uLim = fProjection->GetLimit(ax, 1);
   if (min < dLim) min = dLim*0.98;
   if (max > uLim) max   = uLim*0.98;
}

//______________________________________________________________________________
void TEveProjectionAxesGL::Draw(TGLRnrCtx& rnrCtx) const
{
   // Draw function for TEveProjectionAxesGL. Skips line-pass of outline mode.

   if (rnrCtx.IsDrawPassOutlineLine())
      return;

   TGLObject::Draw(rnrCtx);
}

//______________________________________________________________________________
void TEveProjectionAxesGL::DirectDraw(TGLRnrCtx& rnrCtx) const
{
   // Actual rendering code.
   // Virtual from TGLLogicalShape.

   if (rnrCtx.Selection() || rnrCtx.Highlight()) return;

   glPushAttrib(GL_ENABLE_BIT | GL_POLYGON_BIT);

   glDisable(GL_LIGHTING);
   glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
   glEnable(GL_COLOR_MATERIAL);
   glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
   glDisable(GL_CULL_FACE);

   // Draw on front-clipping plane.
   Float_t old_depth_range[2];
   glGetFloatv(GL_DEPTH_RANGE, old_depth_range);
   glDepthRange(0, 0.001);

   // Frustum size.
   TGLCamera &camera = rnrCtx.RefCamera();
   Float_t l = -camera.FrustumPlane(TGLCamera::kLeft).D();
   Float_t r =  camera.FrustumPlane(TGLCamera::kRight).D();
   Float_t t =  camera.FrustumPlane(TGLCamera::kTop).D();
   Float_t b = -camera.FrustumPlane(TGLCamera::kBottom).D();

   if (fM->fUseColorSet)
   {
       TGLUtil::Color(rnrCtx.ColorSet().Markup());
       fAxisPainter.SetUseAxisColors(kFALSE);
   }

   fProjection = fM->GetManager()->GetProjection();
   glDisable(GL_LIGHTING);
   // Projection center and origin marker.
   {
      Float_t d = ((r-l) > (b-t)) ? (b-t) : (r-l);
      d *= 0.02f;
      if (fM->GetDrawCenter())
      {
         Float_t* c = fProjection->GetProjectedCenter();
         TGLUtil::LineWidth(1);
         glBegin(GL_LINES);
         glVertex3f(c[0] + d, c[1], c[2]); glVertex3f(c[0] - d, c[1], c[2]);
         glVertex3f(c[0], c[1] + d, c[2]); glVertex3f(c[0], c[1] - d, c[2]);
         glVertex3f(c[0], c[1], c[2] + d); glVertex3f(c[0], c[1], c[2] - d);
         glEnd();
      }
      if (fM->GetDrawOrigin())
      {
         TEveVector zero;
         fProjection->ProjectVector(zero, 0);
         TGLUtil::LineWidth(1);
         glBegin(GL_LINES);
         glVertex3f(zero[0] + d, zero[1], zero[2]); glVertex3f(zero[0] - d, zero[1], zero[2]);
         glVertex3f(zero[0], zero[1] + d, zero[2]); glVertex3f(zero[0], zero[1] - d, zero[2]);
         glVertex3f(zero[0], zero[1], zero[2] + d); glVertex3f(zero[0], zero[1], zero[2] - d);
         glEnd();
      }
   }

   //
   // Axes.
   {
      using namespace TMath;
      GLint   vp[4];
      glGetIntegerv(GL_VIEWPORT, vp);
      Float_t refLength =  TMath::Sqrt((TMath::Power(vp[2]-vp[0], 2) + TMath::Power(vp[3]-vp[1], 2)));
      Float_t tickLength = TMath::Sqrt((TMath::Power(r-l, 2) + TMath::Power(t-b, 2)));
      fAxisPainter.SetFontMode(TGLFont::kPixmap);
      fAxisPainter.SetLabelFont(rnrCtx, TGLFontManager::GetFontNameFromId(fM->GetLabelFont()),  TMath::CeilNint(refLength*0.02), tickLength*fM->GetLabelSize());

      Float_t min, max;
      // X-axis.
      if (fM->fAxesMode == TEveProjectionAxes::kAll ||
          fM->fAxesMode == TEveProjectionAxes::kHorizontal)
      {
         GetRange(0, l, r, min, max);
         SplitInterval(min, max, 0);

         fAxisPainter.RefDir().Set(1, 0, 0);
         fAxisPainter.RefTMOff(0).Set(0, tickLength, 0);

         // Bottom.
         glPushMatrix();
         glTranslatef( 0, b, 0);
         fAxisPainter.SetLabelAlign(TGLFont::kCenterH, TGLFont::kTop);
         fAxisPainter.RnrLabels();
         fAxisPainter.RnrLines();
         glPopMatrix();

         // Top.
         glPushMatrix();
         glTranslatef( 0, t, 0);
         fAxisPainter.SetLabelAlign(TGLFont::kCenterH, TGLFont::kBottom);
         fAxisPainter.RefTMOff(0).Negate();
         fAxisPainter.RnrLabels();
         fAxisPainter.RnrLines();
         glPopMatrix();
      }

      // Y-axis.
      if (fM->fAxesMode == TEveProjectionAxes::kAll ||
          fM->fAxesMode == TEveProjectionAxes::kVertical)
      {
         GetRange(1, b, t, min, max);
         SplitInterval(min, max, 1);

         fAxisPainter.RefDir().Set(0, 1, 0);
         fAxisPainter.RefTMOff(0).Set(tickLength, 0 , 0);

         // Left.
         glPushMatrix();
         glTranslatef(l, 0, 0);
         fAxisPainter.SetLabelAlign(TGLFont::kLeft, TGLFont::kCenterV);
         fAxisPainter.RnrLabels();
         fAxisPainter.RnrLines();
         glPopMatrix();

         // Right.
         glPushMatrix();
         glTranslatef(r, 0, 0);
         fAxisPainter.SetLabelAlign(TGLFont::kRight, TGLFont::kCenterV);
         fAxisPainter.RefTMOff(0).Negate();
         fAxisPainter.RnrLabels();
         fAxisPainter.RnrLines();
         glPopMatrix();
      }
   }
   glDepthRange(old_depth_range[0], old_depth_range[1]);

   glPopAttrib();
}
