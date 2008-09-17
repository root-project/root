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
void TEveProjectionAxesGL::DrawTickMarks(Bool_t horizontal, Float_t tmSize) const
{
   // Draw tick-marks on the current axis.

   glBegin(GL_LINES);
   for (std::list<TM_t>::iterator it = fTMList.begin(); it != fTMList.end(); ++it)
   {
      if (horizontal)
      {
         glVertex2f((*it).first, 0);
         glVertex2f((*it).first, tmSize);
      }
      else
      {
         glVertex2f(0, (*it).first);
         glVertex2f(tmSize, (*it).first);
      }
   }
   glEnd();
}

//______________________________________________________________________________
void TEveProjectionAxesGL::SplitInterval(Float_t p1, Float_t p2, Int_t ax) const
{
   // Build an array of tick-mark position-value pairs.

   Float_t down = fProjection->GetLimit(ax, kFALSE)*0.95;
   p1 = TMath::Max(p1, down);

   Float_t up = fProjection->GetLimit(ax, kTRUE)*0.95;
   p2 = TMath::Min(p2, up);

   if (fM->GetLabMode() == TEveProjectionAxes::kValue)
   {
      SplitIntervalByVal(fProjection->GetValForScreenPos(ax, p1), fProjection->GetValForScreenPos(ax, p2), ax);
   }
   else if (fM->GetLabMode() == TEveProjectionAxes::kPosition)
   {
      SplitIntervalByPos(p1, p2, ax);
   }
}

//______________________________________________________________________________
void TEveProjectionAxesGL::SplitIntervalByPos(Float_t p1, Float_t p2, Int_t ax) const
{
   // Add tick-marks at equidistant position.

   Float_t step = (p2-p1)/fM->GetNdivisions();

   TEveVector zeroPos;
   fProjection->ProjectVector(zeroPos);
   Float_t p = zeroPos.fX;
   while (p > p1) {
      fTMList.push_back(TM_t(p , fProjection->GetValForScreenPos(ax, p)));
      p -= step;
   }

   p = zeroPos.fX + step;
   while (p < p2) {
      fTMList.push_back(TM_t(p , fProjection->GetValForScreenPos(ax, p)));
      p += step;
   }
}

//______________________________________________________________________________
void TEveProjectionAxesGL::SplitIntervalByVal(Float_t v1, Float_t v2, Int_t ax) const
{
   // Add tick-marks on fixed value step.

   Float_t step = (v2-v1)/fM->GetNdivisions();

   Float_t v = 0.f;
   while (v > v1) {
      fTMList.push_back(TM_t(fProjection->GetScreenVal(ax, v) , v));
      v -= step;
   }

   v = step;
   while (v < v2) {
      fTMList.push_back(TM_t(fProjection->GetScreenVal(ax, v) , v));
      v += step;
   }
}

//______________________________________________________________________________
void TEveProjectionAxesGL::DirectDraw(TGLRnrCtx& rnrCtx) const
{
   // Actual rendering code.
   // Virtual from TGLLogicalShape.

   if (rnrCtx.Selection() || rnrCtx.Highlight()) return;

   fProjection = fM->GetManager()->GetProjection();
   Float_t bbox[6];
   for(Int_t i=0; i<6; i++)
      bbox[i] = fM->GetManager()->GetBBox()[i];

   // horizontal font setup

   Float_t l =  -rnrCtx.GetCamera()->FrustumPlane(TGLCamera::kLeft).D();
   Float_t r =   rnrCtx.GetCamera()->FrustumPlane(TGLCamera::kRight).D();
   Float_t t =   rnrCtx.GetCamera()->FrustumPlane(TGLCamera::kTop).D();
   Float_t b =  -rnrCtx.GetCamera()->FrustumPlane(TGLCamera::kBottom).D();

   TGLFont font;
   Int_t rng;
   GLint    vp[4];
   glGetIntegerv(GL_VIEWPORT, vp);

   if (fM->fAxesMode == TEveProjectionAxes::kAll || TEveProjectionAxes::kVertical)
      rng = vp[3];
   else
      rng = vp[2];

   Int_t fs =  TGLFontManager::GetFontSize(rng*fM->GetLabelSize(), 8, 36);

   rnrCtx.RegisterFont(fs, "arial", TGLFont::kPixmap, font);
   font.PreRender();

   TGLMatrix modview;
   glGetDoublev(GL_MODELVIEW_MATRIX, modview.Arr());
   TGLVertex3 worldRef;
   Float_t tms = rnrCtx.RefCamera().ViewportDeltaToWorld(worldRef, rng*0.01, 0, &modview).Mag();
   Float_t off = 1.5*tms;

   const char* txt;
   Float_t uLim, dLim;
   Float_t start, end;
   Float_t limFac = 0.98;
   //______________________________________________________________________________
   // X-axis

   fTMList.clear();
   if (fM->fAxesMode == TEveProjectionAxes::kAll
       || (fM->fAxesMode == TEveProjectionAxes::kHorizontal))
   {
      dLim = fProjection->GetLimit(0, 0)*limFac;
      uLim = fProjection->GetLimit(0, 1)*limFac;
      start =  (l > dLim) ?  l : dLim;
      end =    (r < uLim) ?  r : uLim;

      SplitInterval(start, end, 0);
      {
         // bottom
         glPushMatrix();
         glTranslatef((bbox[1]+bbox[0])*0.5, b, 0);
         DrawTickMarks(kTRUE, tms);
         for (std::list<TM_t>::iterator it = fTMList.begin(); it != fTMList.end(); ++it)
         {
            txt =TEveUtil::FormAxisValue((*it).second);
            font.RenderBitmap(txt, (*it).first, +off, 0, TGLFont::kCenterUp);
         }
         glPopMatrix();
      }
      {
         // top
         glPushMatrix();
         glTranslatef((bbox[1]+bbox[0])*0.5, t, 0);
         DrawTickMarks(kTRUE, -tms);
         for (std::list<TM_t>::iterator it = fTMList.begin(); it != fTMList.end(); ++it)
         {
            txt =TEveUtil::FormAxisValue((*it).second);
            font.RenderBitmap(txt, (*it).first, -off, 0, TGLFont::kCenterDown);
         }
         glPopMatrix();
      }
   }

   // Y-axis
   fTMList.clear();
   if (fM->fAxesMode == TEveProjectionAxes::kAll
       || (fM->fAxesMode == TEveProjectionAxes::kVertical))
   {
      // left
      dLim = fProjection->GetLimit(1, 0)*limFac;
      uLim = fProjection->GetLimit(1, 1)*limFac;
      start =  (b > dLim) ? b : dLim;
      end =    (t < uLim) ? t : uLim;
      SplitInterval(start, end, 0);
      {
         glPushMatrix();
         glTranslatef(l, (bbox[3]+bbox[2])*0.5, 0);
         DrawTickMarks(kFALSE, tms);
         for (std::list<TM_t>::iterator it = fTMList.begin(); it != fTMList.end(); ++it)
         {
            txt =TEveUtil::FormAxisValue((*it).second);
            font.RenderBitmap(txt, +off, (*it).first, 0, TGLFont::kRight);
         }
         glPopMatrix();
      }
      // right
      {
         glPushMatrix();
         glTranslatef(r, (bbox[3]+bbox[2])*0.5, 0);
         DrawTickMarks(kFALSE, -tms);
         for (std::list<TM_t>::iterator it = fTMList.begin(); it != fTMList.end(); ++it)
         {
            txt =TEveUtil::FormAxisValue((*it).second);
            font.RenderBitmap(txt, -off, (*it).first, 0,  TGLFont::kLeft);
         }
         glPopMatrix();
      }
   }

   font.PostRender();
}
