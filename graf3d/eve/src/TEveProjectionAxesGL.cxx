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

#include "TGLRnrCtx.h"
#include "TGLIncludes.h"

#include "TMath.h"

//______________________________________________________________________________
//
// OpenGL renderer class for TEveProjectionAxes.
//

ClassImp(TEveProjectionAxesGL);

//______________________________________________________________________________
TEveProjectionAxesGL::TEveProjectionAxesGL() :
   TEveTextGL(),

   fRange(300),
   fLabelSize(0.02),
   fLabelOff(0.5),
   fTMSize(0.02),

   fAxesModel(0),
   fProjection(0)
{
   // Constructor.

   fDLCache    = kFALSE; // Disable display list.
   fMultiColor = kTRUE;
}

/******************************************************************************/

//______________________________________________________________________________
Bool_t TEveProjectionAxesGL::SetModel(TObject* obj, const Option_t* /*opt*/)
{
   // Set model object.
   // Virtual from TGLObject.

   if (SetModelCheckClass(obj, TEveProjectionAxes::Class())) {
      fAxesModel = dynamic_cast<TEveProjectionAxes*>(obj);
      if (fAxesModel->GetManager() == 0)
         return kFALSE;
      TEveTextGL::fM = fAxesModel;
      return kTRUE;
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

/******************************************************************************/

//______________________________________________________________________________
void TEveProjectionAxesGL::RenderText(const char* txt, Float_t x, Float_t y) const
{
   // Render FTFont at given location.

   if (fFont.GetMode() < TGLFont::kTexture) {
      glRasterPos3f(0, 0, 0);
      glBitmap(0, 0, 0, 0, x, y, 0);
      fFont.Render(txt);
   } else {
      glPushMatrix();
      glTranslatef(x, y, 0);
      fFont.Render(txt);
      glPopMatrix();
   }
}

/******************************************************************************/

//______________________________________________________________________________
void TEveProjectionAxesGL::DrawTickMarks(Float_t y) const
{
   // Draw tick-marks on the current axis.

   if (fFont.GetMode() == TGLFont::kTexture) glDisable(GL_TEXTURE_2D);

   glBegin(GL_LINES);
   for (std::list<TM_t>::iterator it = fTMList.begin(); it != fTMList.end(); ++it)
   {
      glVertex2f((*it).first, 0);
      glVertex2f((*it).first, y);
   }
   glEnd();

   if (fFont.GetMode() == TGLFont::kTexture) glEnable(GL_TEXTURE_2D);
}

//______________________________________________________________________________
void TEveProjectionAxesGL::DrawHInfo() const
{
   // Draw labels on horizontal axis.

   Float_t tmH = fTMSize*fRange;
   DrawTickMarks(-tmH);

   Float_t off = tmH + fLabelOff*tmH;
   Float_t llx, lly, llz, urx, ury, urz;
   const char* txt;
   for (std::list<TM_t>::iterator it = fTMList.begin(); it != fTMList.end(); ++it)
   {
      glPushMatrix();
      glTranslatef((*it).first, -off, 0);
      txt = TEveUtil::FormAxisValue((*it).second);
      fFont.BBox(txt, llx, lly, llz, urx, ury, urz);
      Float_t xd = -0.5f*(urx+llx);
      if (txt[0] == '-')
         xd -= 0.5f * (urx-llx) / strlen(txt);
      RenderText(txt, xd, -ury);
      glPopMatrix();
   }

   fTMList.clear();
}

//______________________________________________________________________________
void TEveProjectionAxesGL::DrawVInfo() const
{
   // Draw labels on vertical axis.

   Float_t tmH = fTMSize*fRange;

   glPushMatrix();
   glRotatef(90, 0, 0, 1);
   DrawTickMarks(tmH);
   glPopMatrix();

   Float_t off = fLabelOff*tmH + tmH;
   Float_t llx, lly, llz, urx, ury, urz;
   const char* txt;
   for (std::list<TM_t>::iterator it = fTMList.begin(); it!= fTMList.end(); ++it)
   {
      glPushMatrix();
      glTranslatef(-off, (*it).first, 0);
      txt = TEveUtil::FormAxisValue((*it).second);
      fFont.BBox(txt, llx, lly, llz, urx, ury, urz);
      RenderText(txt, -urx, -0.38f*(ury+lly));
      glPopMatrix();
   }

   fTMList.clear();
}

/******************************************************************************/

//______________________________________________________________________________
void TEveProjectionAxesGL::SplitInterval(Float_t p1, Float_t p2, Int_t ax) const
{
   // Build an array of tick-mark position-value pairs.

   Float_t down = fProjection->GetLimit(ax, kFALSE)*0.95;
   p1 = TMath::Max(p1, down);

   Float_t up = fProjection->GetLimit(ax, kTRUE)*0.95;
   p2 = TMath::Min(p2, up);

   if (fAxesModel->GetStepMode() == TEveProjectionAxes::kValue)
   {
      SplitIntervalByVal(fProjection->GetValForScreenPos(ax, p1), fProjection->GetValForScreenPos(ax, p2), ax);
   }
   else if (fAxesModel->GetStepMode() == TEveProjectionAxes::kPosition)
   {
      SplitIntervalByPos(p1, p2, ax);
   }
}

//______________________________________________________________________________
void TEveProjectionAxesGL::SplitIntervalByPos(Float_t p1, Float_t p2, Int_t ax) const
{
   // Add tick-marks at equidistant position.

   TEveVector zeroPos;
   fProjection->ProjectVector(zeroPos);
   Float_t step = (p2-p1)/fAxesModel->GetNumTickMarks();

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

   Float_t step = (v2-v1)/fAxesModel->GetNumTickMarks();

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

/******************************************************************************/

//______________________________________________________________________________
void TEveProjectionAxesGL::DirectDraw(TGLRnrCtx& rnrCtx) const
{
   // Actual rendering code.
   // Virtual from TGLLogicalShape.

   fProjection = fAxesModel->GetManager()->GetProjection();
   Float_t* bbox = fAxesModel->GetBBox();
   fRange = bbox[1] - bbox[0];
   TEveVector zeroPos;
   fProjection->ProjectVector(zeroPos);

   // axes lines
   glBegin(GL_LINES);
   glVertex3f(bbox[0], bbox[2], 0.);
   glVertex3f(bbox[1], bbox[2], 0.);
   glVertex3f(bbox[0], bbox[2], 0.);
   glVertex3f(bbox[0], bbox[3], 0.);
   glEnd();

   // projection center and origin marker
   Float_t d = 10;
   if (fAxesModel->GetDrawCenter()) {
      Float_t* c = fProjection->GetProjectedCenter();
      TGLUtil::Color3f(1., 0., 0.);
      glBegin(GL_LINES);
      glVertex3f(c[0] +d, c[1],    c[2]);     glVertex3f(c[0] - d, c[1]   , c[2]);
      glVertex3f(c[0] ,   c[1] +d, c[2]);     glVertex3f(c[0]    , c[1] -d, c[2]);
      glVertex3f(c[0] ,   c[1],    c[2] + d); glVertex3f(c[0]    , c[1]   , c[2] - d);
      glEnd();
   }
   if (fAxesModel->GetDrawOrigin()) {
      TEveVector zero;
      fProjection->ProjectVector(zero);
      TGLUtil::Color3f(1., 1., 1.);
      glBegin(GL_LINES);
      glVertex3f(zero[0] + d, zero[1],     zero[2]);     glVertex3f(zero[0] - d, zero[1]   ,  zero[2]);
      glVertex3f(zero[0] ,    zero[1] + d, zero[2]);     glVertex3f(zero[0]    , zero[1] - d, zero[2]);
      glVertex3f(zero[0] ,    zero[1],     zero[2] + d); glVertex3f(zero[0]    , zero[1]   ,  zero[2] - d);
      glEnd();
   }


   SetFont(rnrCtx);
   fFont.PreRender(fAxesModel->GetAutoLighting(), fAxesModel->GetLighting());

   // title
   glPushMatrix();
   glTranslatef(zeroPos.fX, bbox[3]*1.1, 0);
   Float_t llx, lly, llz, urx, ury, urz;
   fFont.BBox(fAxesModel->GetText(), llx, lly, llz, urx, ury, urz);
   RenderText(fAxesModel->GetText(), -llx, 0);
   glPopMatrix();

   // X-axis tick-marks & labels
   glPushMatrix();
   glTranslatef(0, bbox[2], 0);
   SplitInterval(bbox[0], bbox[1], 0);
   DrawHInfo();
   glPopMatrix();

   // Y-axis tick-marks & labels
   glPushMatrix();
   glTranslatef(bbox[0], 0, 0);
   SplitInterval(bbox[2], bbox[3], 1);
   DrawVInfo();
   glPopMatrix();

   fFont.PostRender();

   fProjection = 0;
}
