// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveProjectionManagerGL.h"
#include "TEveProjectionManager.h"

#include "TGLRnrCtx.h"
#include "TGLIncludes.h"
#include "TGLText.h"
#include "TMath.h"

#include <list>

//______________________________________________________________________________
// TEveProjectionManagerGL
//
// GL-renderer for TEveProjectionManager.

ClassImp(TEveProjectionManagerGL)

//______________________________________________________________________________
TEveProjectionManagerGL::TEveProjectionManagerGL() :
   TGLObject(),

   fRange(300),
   fLabelSize(0.02),
   fLabelOff(0.018),
   fTMSize(0.02),

   fM(0),
   fText(0)
{
   // Constructor.

   fDLCache = kFALSE; // Disable display list.
   fText = new TGLText();
   fText->SetGLTextFont(40);
   fText->SetTextColor(0);
}

/******************************************************************************/

//______________________________________________________________________________
const char* TEveProjectionManagerGL::GetText(Float_t x) const
{
   // Get formatted text.

   using  namespace TMath;
   if (Abs(x) > 1000) {
      Float_t v = 10*TMath::Nint(x/10.0f);
      return Form("%.0f", v);
   } else if(Abs(x) > 100) {
      Float_t v = TMath::Nint(x);
      return Form("%.0f", v);
   } else if (Abs(x) > 10)
      return Form("%.1f", x);
   else if ( Abs(x) > 1 )
      return Form("%.2f", x);
   else
      return Form("%.3f", x);
}

/******************************************************************************/

//______________________________________________________________________________
void TEveProjectionManagerGL::SetRange(Float_t pos, Int_t ax) const
{
   // Set values for bounding box.
  
   using namespace TMath;
   Float_t limit =  fM->GetProjection()->GetLimit(ax, pos > 0 ? kTRUE: kFALSE);
   if (fM->GetProjection()->GetDistortion() > 0.001 && Abs(pos) > Abs(limit *0.97))
   {
      fPos.push_back(limit *0.7);
      fVals.push_back(fM->GetProjection()->GetValForScreenPos(ax, fPos.back()));
   }
   else
   {
      fPos.push_back(pos);
      fVals.push_back(fM->GetProjection()->GetValForScreenPos(ax, fPos.back()));
   }
}

/******************************************************************************/

//______________________________________________________________________________
void TEveProjectionManagerGL::DrawTickMarks(Float_t tm) const
{
   // Draw tick-marks on the current axis.

   glBegin(GL_LINES);
   for (std::list<Float_t>::iterator pi = fPos.begin(); pi!= fPos.end(); ++pi)
   {
      glVertex3f(*pi, 0,   0.);
      glVertex3f(*pi, tm, 0.);
   }
   glEnd();
}

//______________________________________________________________________________
void TEveProjectionManagerGL::DrawHInfo() const
{
   // Draw labels on horizontal axis.

   Float_t tms = fTMSize*fRange;
   DrawTickMarks(-tms);

   glPushMatrix();
   glRotatef(-90, 1, 0, 0);
   glTranslatef(0, 0, -tms -fLabelOff*fRange);
   const char* txt;
   Float_t llx, lly, llz, urx, ury, urz;
   std::list<Float_t>::iterator vi = fVals.begin();
   for( std::list<Float_t>::iterator pi = fPos.begin(); pi!= fPos.end(); pi++)
   {
      txt = GetText(*vi);
      fText->BBox(txt, llx, lly, llz, urx, ury, urz);
      fText->PaintGLText(*pi -(urx-llx)*fText->GetTextSize()*0.5, 0, 0, txt);
      vi++;
   }
   glPopMatrix();

   fPos.clear(); fVals.clear();
}

//______________________________________________________________________________
void TEveProjectionManagerGL::DrawVInfo() const
{
   // Draw labels on vertical axis.

   Float_t tms = fTMSize*fRange;
   glRotatef(90, 0, 0, 1);
   DrawTickMarks(tms);
   glRotatef(-90, 0, 0, 1);

   glPushMatrix();
   glRotatef(-90, 1, 0, 0);
   glTranslatef(-fLabelOff*fRange -tms, 0, 0);
   const char* txt;
   Float_t llx, lly, llz, urx, ury, urz;
   std::list<Float_t>::iterator vi = fVals.begin();
   for (std::list<Float_t>::iterator pi = fPos.begin(); pi!= fPos.end(); ++pi)
   {
      txt= GetText(*vi);
      fText->BBox(txt, llx, lly, llz, urx, ury, urz);
      fText->PaintGLText(-(urx-llx)*fText->GetTextSize(), 0, *pi - (ury - lly)*fText->GetTextSize()*0.5, txt);
      vi++;
   }
   glPopMatrix();

   fPos.clear(); fVals.clear();
}

/******************************************************************************/

//______________________________________________________________________________
void TEveProjectionManagerGL::SplitInterval(Int_t ax) const
{
   // Build a list of labels and their positions.

   if (fM->GetSplitInfoLevel())
   {
      if(fM->GetSplitInfoMode())
         SplitIntervalByVal(fVals.front(), fVals.back(), ax, 0);
      else
         SplitIntervalByPos(fPos.front(), fPos.back(), ax, 0);
   }
}

//______________________________________________________________________________
void TEveProjectionManagerGL::SplitIntervalByPos(Float_t minp, Float_t maxp, Int_t ax, Int_t level) const
{
   // Add tick-mark and label with position in the middle of given interval.

   Float_t p = (minp+maxp)*0.5;
   fPos.push_back(p);
   Float_t v = fM->GetProjection()->GetValForScreenPos(ax, p);
   fVals.push_back(v);
   level++;
   if (level<fM->GetSplitInfoLevel())
   {
      SplitIntervalByPos(minp, p , ax, level);
      SplitIntervalByPos(p, maxp, ax, level);
   }
}

//______________________________________________________________________________
void TEveProjectionManagerGL::SplitIntervalByVal(Float_t minv, Float_t maxv, Int_t ax, Int_t level) const
{
   // Add tick-mark and label with value in the middle of given interval.

   Float_t v = (minv+maxv)*0.5;
   fVals.push_back(v);
   Float_t p = fM->GetProjection()->GetScreenVal(ax, v);
   fPos.push_back(p);
   level++;
   if (level<fM->GetSplitInfoLevel())
   {
      SplitIntervalByVal(minv, v , ax, level);
      SplitIntervalByVal(v, maxv, ax, level);
   }
}

/******************************************************************************/

//______________________________________________________________________________
void TEveProjectionManagerGL::DirectDraw(TGLRnrCtx & /*rnrCtx*/) const
{
   // Actual rendering code.
   // Virtual from TGLLogicalShape.

   GLboolean lightp;
   glGetBooleanv(GL_LIGHTING, &lightp);
   if (lightp) glDisable(GL_LIGHTING);

   Float_t* bbox = fM->GetBBox();
   fRange = bbox[1] - bbox[0];
   // printf("bbox %f, %f\n", bbox[0], bbox[1]);
   TEveVector zeroPos;
   fM->GetProjection()->ProjectVector(zeroPos);
   fText->SetTextSize(fLabelSize*fRange);
   fText->SetTextColor(fM->GetAxisColor());

   { // horizontal
      glPushMatrix();
      glTranslatef(0, bbox[2], 0);
      // left
      SetRange(bbox[0], 0);
      fPos.push_back(zeroPos.fX); fVals.push_back(0);
      SplitInterval(0);
      DrawHInfo();
      // right
      fPos.push_back(zeroPos.fX); fVals.push_back(0);
      SetRange(bbox[1], 0);
      SplitInterval(0); fVals.pop_front(); fPos.pop_front();
      DrawHInfo();
      glPopMatrix();
   }
   { // vertical
      glPushMatrix();
      glTranslatef(bbox[0], 0, 0);
      // bottom
      fPos.push_back(zeroPos.fY);fVals.push_back(0);
      SetRange(bbox[2], 1);
      SplitInterval(1);
      DrawVInfo();
      // top
      fPos.push_back(zeroPos.fY); fVals.push_back(0);
      SetRange(bbox[3], 1);
      SplitInterval(1);fPos.pop_front(); fVals.pop_front();
      DrawVInfo();
      glPopMatrix();
   }

   // body
   glBegin(GL_LINES);
   glVertex3f(bbox[0], bbox[2], 0.);
   glVertex3f(bbox[1], bbox[2], 0.);
   glVertex3f(bbox[0], bbox[2], 0.);
   glVertex3f(bbox[0], bbox[3], 0.);
   glEnd();

   Float_t d = 10;
   if (fM->GetDrawCenter())
   {
      Float_t* c = fM->GetProjection()->GetProjectedCenter();
      TGLUtil::Color3f(1., 0., 0.);
      glBegin(GL_LINES);
      glVertex3f(c[0] +d, c[1],    c[2]);     glVertex3f(c[0] - d, c[1]   , c[2]);
      glVertex3f(c[0] ,   c[1] +d, c[2]);     glVertex3f(c[0]    , c[1] -d, c[2]);
      glVertex3f(c[0] ,   c[1],    c[2] + d); glVertex3f(c[0]    , c[1]   , c[2] - d);
      glEnd();

   }

   if (fM->GetDrawOrigin())
   {
      TEveVector zero;
      fM->GetProjection()->ProjectVector(zero);
      TGLUtil::Color3f(1., 1., 1.);
      glBegin(GL_LINES);
      glVertex3f(zero[0] +d, zero[1],    zero[2]);     glVertex3f(zero[0] - d, zero[1]   , zero[2]);
      glVertex3f(zero[0] ,   zero[1] +d, zero[2]);     glVertex3f(zero[0]    , zero[1] -d, zero[2]);
      glVertex3f(zero[0] ,   zero[1],    zero[2] + d); glVertex3f(zero[0]    , zero[1]   , zero[2] - d);
      glEnd();
   }
   if (lightp) glEnable(GL_LIGHTING);
}


/******************************************************************************/

//______________________________________________________________________________
Bool_t TEveProjectionManagerGL::SetModel(TObject* obj, const Option_t* /*opt*/)
{
   // Set model object.
   // Virtual from TGLObject.

   if (SetModelCheckClass(obj, TEveProjectionManager::Class())) {
      fM = dynamic_cast<TEveProjectionManager*>(obj);
      return kTRUE;
   }
   return kFALSE;
}

//______________________________________________________________________________
void TEveProjectionManagerGL::SetBBox()
{
   // Fill the bounding-box data of the logical-shape.
   // Virtual from TGLObject.

   // !! This ok if master sub-classed from TAttBBox
   SetAxisAlignedBBox(((TEveProjectionManager*)fExternalObj)->AssertBBox());
}
