// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLAxisPainter.h"

#include "TGLRnrCtx.h"
#include "TGLCamera.h"
#include "TGLIncludes.h"
#include "TGLRnrCtx.h"
#include "TGLFontManager.h"
#include "TAxis.h"
#include "THLimitsFinder.h"

#include "TMath.h"


//______________________________________________________________________________
// Axsis attributes reguired to be drawn in GL.
//

ClassImp(TGLAxisAttrib);

//______________________________________________________________________________
TGLAxisAttrib::TGLAxisAttrib() :
   TAttAxis(),

   fDir(1, 0, 0),
   fMin(0),
   fMax(100),

   fTMNDim(1),

   fTextAlign(kCenterDown),

   fRelativeFontSize(kFALSE),
   fAbsLabelFontSize(24),
   fAbsTitleFontSize(24),

   fLabelFontName("arial"),
   fTitleFontName("arial")

{
   // Constructor.

   fNdivisions = 510;
   fLabelSize = 0.04;

   fLabelColor = kWhite;
   fTitleColor = kWhite;

   fTMScale[0] = 1;
   fTMScale[1] = 0.5;
   fTMScale[2] = 0.25;
}


//______________________________________________________________________________
//
// Utility class to paint axis in GL.


ClassImp(TGLAxisPainter);

//______________________________________________________________________________
const char* TGLAxisPainter::FormAxisValue(Float_t x) const
{
   // Returns formatted text suitable for display of value.

   using namespace TMath;

   if (Abs(x) > 1000)
      return Form("%d", (Int_t) 10*Nint(x/10.0f));
   if (Abs(x) > 100 || x == Nint(x))
      return Form("%d", (Int_t) Nint(x));
   if (Abs(x) > 10)
      return Form("%.1f", x);
   if (Abs(x) >= 0.01 )
      return Form("%.2f", x);
   return "0";
}

//______________________________________________________________________________
inline void TGLAxisPainter::DrawTick(TGLVector3 &tv, Int_t order) const
{
   // Draw tick-marks in supprted dimensions.

   for (Int_t dim=0; dim < fAtt->fTMNDim; dim++)
   {
      glVertex3dv(tv.Arr());
      glVertex3dv((tv+fAtt->fTMOff[dim]*fAtt->fTMScale[order]).Arr());
   }
}

//______________________________________________________________________________
void TGLAxisPainter::RnrText(const char* txt, TGLVector3 pos, TGLFont &font) const
{
   // Render text at the given position. Make offset depending of text aligment.

   glPushMatrix();
   glTranslatef(pos.X(), pos.Y(), pos.Z());

   Float_t llx, lly, llz, urx, ury, urz;
   font.BBox(txt, llx, lly, llz, urx, ury, urz);
   if (txt[0] == '-')
      urx += (urx-llx)/strlen(txt);

   Float_t x=0, y=0;

   switch (fAtt->fTextAlign)
   {
      case TGLAxisAttrib::kCenterDown:
         x = -urx*0.5; y = -ury;
         break;
      case TGLAxisAttrib::kCenterUp:
         x = -urx; y = 0;
         break;
      case TGLAxisAttrib::kLeft:
         x = -urx; y =(lly -ury)*0.5;
         break;
      case TGLAxisAttrib::kRight:
         x = 0; y = -ury*0.5;
         break;
      default:
         break;
   };

   glRasterPos2i(0, 0);
   glBitmap(0, 0, 0, 0, x, y, 0);
   font.Render(txt);

   glPopMatrix();
}

//______________________________________________________________________________
void TGLAxisPainter::Paint(TGLRnrCtx &rnrCtx, TGLAxisAttrib &att)
{
   // Paint axis body, tickmarks and labels.

   if (rnrCtx.Selection() || rnrCtx.Highlight())
      return;

   fAtt = &att;

   TGLVector3 start = att.fDir*att.fMin;
   TGLVector3 end = att.fDir*att.fMax;

   // optimise
   Int_t n1a = TMath::FloorNint(att.fNdivisions/100);
   Int_t n2a = att.fNdivisions-n1a*100;
   Int_t bn1, bn2;
   Double_t bw1, bw2; // bin with
   Double_t bl1, bh1, bl2, bh2; // bin low, high

   THLimitsFinder::Optimize(att.fMin, att.fMax, n1a, bl1, bh1, bn1, bw1);
   THLimitsFinder::Optimize(bl1, bl1+bw1, n2a, bl2, bh2, bn2, bw2);

   /**************************************************************************/

   TGLFont font;
   Double_t len=0;
   if (att.fRelativeFontSize)
   {
      GLdouble mm[16];
      GLdouble pm[16];
      GLint    vp[4];
      glGetDoublev(GL_MODELVIEW_MATRIX,  mm);
      glGetDoublev(GL_PROJECTION_MATRIX, pm);
      glGetIntegerv(GL_VIEWPORT, vp);

      GLdouble dn[3];
      GLdouble up[3];
      gluProject(start.X(), start.Y(), start.Z(), mm, pm, vp, &dn[0], &dn[1], &dn[2]);
      gluProject(end.X(), end.Y(), end.Z(), mm, pm, vp, &up[0], &up[1], &up[2]);
      len = TMath::Sqrt((  up[0]-dn[0])*(up[0]-dn[0])
                        + (up[1]-dn[1])*(up[1]-dn[1])
                        + (up[2]-dn[2])*(up[2]-dn[2]));
   }

   // labels
   {
      Int_t fs = att.fRelativeFontSize ? Int_t(att.GetLabelSize()*len):att.fAbsLabelFontSize;
      att.fAbsLabelFontSize = TGLFontManager::GetFontSize(fs, 12, 36);
      rnrCtx.RegisterFont(att.fAbsLabelFontSize, att.fLabelFontName.Data(), TGLFont::kPixmap, font);

      TGLUtil::Color(att.fLabelColor);
      glPushMatrix();
      TGLVector3 off = (att.fTMOff[0])*2.5; // tmp
      glTranslated (off.X(), off.Y(), off.Z());

      font.PreRender();
      TGLVector3 pos  = att.fDir*bl1;
      TGLVector3 step = att.fDir*bw1;
      for (Int_t i=0; i<=bn1; i++)
      {
         RnrText(FormAxisValue(bl1+i*bw1), pos, font);
         pos += step;
      }
      font.PostRender();
      glPopMatrix();
      rnrCtx.ReleaseFont(font);
   }


   // title
   if (att.fTitle.Length())
   {
      Int_t fs = (att.fRelativeFontSize)? Int_t(att.GetTitleSize()*len) : att.fAbsTitleFontSize;

      rnrCtx.RegisterFont(TGLFontManager::GetFontSize(fs, 12, 36),
                          att.fTitleFontName.Data(), TGLFont::kPixmap, font);
      TGLUtil::Color(att.fTitleColor);
      font.PreRender();
      TGLVector3 pos = att.fTitlePos;
      pos  += att.fTMOff[0]*2.5;
      RnrText(att.fTitle.Data(), pos, font);
      font.PostRender();
      rnrCtx.ReleaseFont(font);
   }

   /**************************************************************************/

   TGLUtil::Color(att.fAxisColor);
   glBegin(GL_LINES);

   // body
   {
      glVertex3dv(start.Arr());
      glVertex3dv(end.Arr());
   }

   // tick-marks
   {
      TGLVector3 tmStep1 = att.fDir*bw1;
      TGLVector3 tmStep2 = att.fDir*bw2;
      TGLVector3 tv1 = att.fDir*bl1;
      TGLVector3 tv2;
      for (Int_t t1=0; t1<bn1; t1++)
      {
         DrawTick(tv1, 0);
         tv2 = tv1 + att.fDir*(bl2-bl1);
         for (Int_t t2=0; t2<=bn2; t2++)
         {
            DrawTick(tv2, 1);
            tv2 += tmStep2;
         }
         tv1 += tmStep1;
      }

      // complete last
      DrawTick(tv1, 0);

      // complete up edges for first order
      Int_t nc = Int_t((att.fMax-bh1)/bw2);
      tv2 = att.fDir*bh1;
      for(Int_t t2=0; t2<=nc; t2++)
      {
         DrawTick(tv2, 1);
         tv2 += tmStep2;
      }
      // complete low edges for first order
      nc = Int_t((bl1-att.fMin)/bw2);
      tv2 = att.fDir*bl1;
      for(Int_t t2=0; t2<=nc; t2++)
      {
         DrawTick(tv2, 1);
         tv2 -= tmStep2;
      }
   }
   glEnd();
}
