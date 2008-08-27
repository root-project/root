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

#include "TMath.h"

//______________________________________________________________________________
//
// OpenGL renderer class for TEveProjectionAxes.
//

ClassImp(TEveProjectionAxesGL);

//______________________________________________________________________________
TEveProjectionAxesGL::TEveProjectionAxesGL() :
   TGLObject(),

   fTMSize(0),

   fM(0),
   fProjection(0)
{
   // Constructor.

   fDLCache    = kFALSE; // Disable display list.
}

/******************************************************************************/

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

/******************************************************************************/

//______________________________________________________________________________
void TEveProjectionAxesGL::DrawTickMarks(Float_t y) const
{
   // Draw tick-marks on the current axis.

   glBegin(GL_LINES);
   for (std::list<TM_t>::iterator it = fTMList.begin(); it != fTMList.end(); ++it)
   {
      glVertex2f((*it).first, 0);
      glVertex2f((*it).first, y);
   }
   glEnd();

}

/******************************************************************************/

//______________________________________________________________________________
void TEveProjectionAxesGL::RenderText(const char* txt, Float_t x, Float_t y, TGLFont &font) const
{
   // Render FTFont at given location.

   if (font.GetMode() < TGLFont::kTexture) {
      glRasterPos3f(0, 0, 0);
      glBitmap(0, 0, 0, 0, x, y, 0);
      font.Render(txt);
   } else {
      glPushMatrix();
      glTranslatef(x, y, 0);
      font.Render(txt);
      glPopMatrix();
   }
}

//______________________________________________________________________________
void TEveProjectionAxesGL::DrawHInfo(TGLFont &font) const
{
   // Draw labels on horizontal axis.

   DrawTickMarks(-fTMSize);

   Float_t off = 2.5*fTMSize;
   Float_t llx, lly, llz, urx, ury, urz;
   const char* txt;
   for (std::list<TM_t>::iterator it = fTMList.begin(); it != fTMList.end(); ++it)
   {
      glPushMatrix();
      glTranslatef((*it).first, -off, 0);
      txt = TEveUtil::FormAxisValue((*it).second);
      font.BBox(txt, llx, lly, llz, urx, ury, urz);
      Float_t xd = -0.5f*(urx+llx);
      if (txt[0] == '-')
         xd -= 0.5f * (urx-llx) / strlen(txt);
      RenderText(txt,  xd, -ury, font);
      glPopMatrix();
   }

   fTMList.clear();
}

//______________________________________________________________________________
void TEveProjectionAxesGL::DrawVInfo(TGLFont &font) const
{
   // Draw labels on vertical axis.

   glPushMatrix();
   glRotatef(90, 0, 0, 1);
   DrawTickMarks(fTMSize);
   glPopMatrix();

   Float_t off = 2.5*fTMSize;
   Float_t llx, lly, llz, urx, ury, urz;
   const char* txt;
   for (std::list<TM_t>::iterator it = fTMList.begin(); it!= fTMList.end(); ++it)
   {
      glPushMatrix();
      glTranslatef(-off, (*it).first, 0);
      txt = TEveUtil::FormAxisValue((*it).second);
      font.BBox(txt, llx, lly, llz, urx, ury, urz);
      RenderText(txt, -urx, -0.38f*(ury+lly), font);
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

   TEveVector zeroPos;
   fProjection->ProjectVector(zeroPos);
   Float_t step = (p2-p1)/fM->GetNdiv();

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

   Float_t step = (v2-v1)/fM->GetNdiv();

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

   if (rnrCtx.Selection() || rnrCtx.Highlight()) return;

   fProjection = fM->GetManager()->GetProjection();
   Float_t bbox[6];
   for(Int_t i=0; i<6; i++)
      bbox[i] = fM->GetManager()->GetBBox()[i];

   bbox[0] = (fM->fBoxOffsetX+1)*bbox[0];
   bbox[1] = (fM->fBoxOffsetX+1)*bbox[1];
   bbox[2] = (fM->fBoxOffsetY+1)*bbox[2];
   bbox[3] = (fM->fBoxOffsetY+1)*bbox[3];



   fTMSize = (bbox[1] -bbox[0])*0.02;
   TEveVector zeroPos;
   fProjection->ProjectVector(zeroPos);

   //horizontal font setup
   GLdouble mm[16];
   GLdouble pm[16];
   GLint    vp[4];
   glGetDoublev(GL_MODELVIEW_MATRIX,  mm);
   glGetDoublev(GL_PROJECTION_MATRIX, pm);
   glGetIntegerv(GL_VIEWPORT, vp);

   TGLVector3 start(bbox[0], 0, 0);
   TGLVector3 end(bbox[1], 0, 0);
   GLdouble dn[3];
   GLdouble up[3];
   gluProject(start.X(), start.Y(), start.Z(), mm, pm, vp, &dn[0], &dn[1], &dn[2]);
   gluProject(end.X(), end.Y(), end.Z(), mm, pm, vp, &up[0], &up[1], &up[2]);
   Float_t rng = TMath::Sqrt((  up[0]-dn[0])*(up[0]-dn[0])
                             + (up[1]-dn[1])*(up[1]-dn[1])
                             + (up[2]-dn[2])*(up[2]-dn[2]));

   TGLFont font;
   Int_t fs =  TGLFontManager::GetFontSize(rng*fM->fLabelSize, 8, 36);
   rnrCtx.RegisterFont(fs, "arial", TGLFont::kPixmap, font);

 glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT);
 glLineWidth(1.);

   // X-axis
   if (fM->fAxesMode == TEveProjectionAxes::kAll
       || (fM->fAxesMode == TEveProjectionAxes::kHorizontal))
   {
      glBegin(GL_LINES);
      glVertex3f(bbox[0], bbox[2], 0.);
      glVertex3f(bbox[1], bbox[2], 0.);
      glEnd();

      glPushMatrix();
      glTranslatef(0, bbox[2], 0);
      SplitInterval(bbox[0], bbox[1], 0);
      font.PreRender();
      DrawHInfo(font);
      font.PostRender();
      glPopMatrix();
   }

   //______________________________________________________________________________

   // Y-axis
   if (fM->fAxesMode == TEveProjectionAxes::kAll
       || (fM->fAxesMode == TEveProjectionAxes::kVertical))
   {
      glBegin(GL_LINES);
      glVertex3f(bbox[0], bbox[2], 0.);
      glVertex3f(bbox[0], bbox[3], 0.);
      glEnd();

      font.PreRender();
      glPushMatrix();
      glTranslatef(bbox[0], 0, 0);
      SplitInterval(bbox[2], bbox[3], 1);
      DrawVInfo(font);
      glPopMatrix();
      font.PostRender();
   }

   // title
   glPushMatrix();
   glTranslatef(zeroPos.fX, bbox[3]*1.1, 0);
   Float_t llx, lly, llz, urx, ury, urz;
   font.BBox(fM->GetTitle(), llx, lly, llz, urx, ury, urz);
   RenderText(fM->GetTitle(), -llx, 0, font);
   glPopMatrix();


   // projection center and origin marker
   Float_t d = 10;
   if (fM->GetDrawCenter()) {
      Float_t* c = fProjection->GetProjectedCenter();
      TGLUtil::Color3f(1., 0., 0.);
      glBegin(GL_LINES);
      glVertex3f(c[0] +d, c[1],    c[2]);     glVertex3f(c[0] - d, c[1]   , c[2]);
      glVertex3f(c[0] ,   c[1] +d, c[2]);     glVertex3f(c[0]    , c[1] -d, c[2]);
      glVertex3f(c[0] ,   c[1],    c[2] + d); glVertex3f(c[0]    , c[1]   , c[2] - d);
      glEnd();
   }
   if (fM->GetDrawOrigin()) {
      TEveVector zero;
      fProjection->ProjectVector(zero);
      TGLUtil::Color3f(1., 1., 1.);
      glBegin(GL_LINES);
      glVertex3f(zero[0] + d, zero[1],     zero[2]);     glVertex3f(zero[0] - d, zero[1]   ,  zero[2]);
      glVertex3f(zero[0] ,    zero[1] + d, zero[2]);     glVertex3f(zero[0]    , zero[1] - d, zero[2]);
      glVertex3f(zero[0] ,    zero[1],     zero[2] + d); glVertex3f(zero[0]    , zero[1]   ,  zero[2] - d);
      glEnd();
   }

   glPopAttrib();
}
