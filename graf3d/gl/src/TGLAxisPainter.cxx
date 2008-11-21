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
// Axis attributes required to be drawn in GL.
//

ClassImp(TGLAxisAttrib);

//______________________________________________________________________________
TGLAxisAttrib::TGLAxisAttrib() :
   TAttAxis(),

   fDir(1, 0, 0),
   fMin(0),
   fMax(100),

   fTMNDim(1),

   fTextAlign(TGLFont::kCenterDown),

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
TGLAxisPainter::TGLAxisPainter():
   fAtt(0),

   fMaxDigits(5),
   fDecimals(0),
   fExp(0)
{
   // Constructor.

}

//______________________________________________________________________________
void TGLAxisPainter::LabelsLimits(const char *label, Int_t &first, Int_t &last) const
{
   // Find first and last character of a label.

   last = strlen(label)-1;
   for (Int_t i=0; i<=last; i++) {
      if (strchr("1234567890-+.", label[i]) ) { first = i; return; }
   }
   Error("LabelsLimits", "attempt to draw a blank label");
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
void TGLAxisPainter::FormAxisValue(Float_t wlabel, char* label) const
{
   // Returns formatted text suitable for display of value.

   sprintf(label,&fFormat[0],wlabel);
   Int_t first, last;
   LabelsLimits(label, first, last);

   char chtemp[256];
   if (label[first] == '.') { //check if '.' is preceeded by a digit
      strcpy(chtemp, "0");
      strcat(chtemp, &label[first]);
      strcpy(label, chtemp);
      first = 1; last = strlen(label);
   }
   if (label[first] == '-' && label[first+1] == '.') {
      strcpy(chtemp, "-0");
      strcat(chtemp, &label[first+1]);
      strcpy(label, chtemp);
      first = 1; last = strlen(label);
   }

   //  We eliminate the non significant 0 after '.'
   if (fDecimals) {
      char *adot = strchr(label,'.');
      if (adot) adot[fDecimals] = 0;
   } else {
      while (label[last] == '0') { label[last] = 0; last--;}
   }
   // We eliminate the dot, unless dot is forced.
   if (label[last] == '.') {
      label[last] = 0; last--;
   }

   //  Make sure the label is not "-0"
   if (last-first == 1 && label[first] == '-' && label[last]  == '0') {
      strcpy(label, "0");
      label[last] = 0;
   }
}

//______________________________________________________________________________
void TGLAxisPainter::SetTextFormat(Double_t bw1)
{
   // Construct print format from given primary bin width.

   Double_t absMax = TMath::Max(TMath::Abs(fAtt->fMin),TMath::Abs(fAtt->fMax));
   Double_t epsilon = 1e-5;
   Double_t absMaxLog = TMath::Log10(absMax) + epsilon;

   fExp   = 0;
   Int_t if1, if2;
   Double_t xmicros = TMath::Power(10,-fMaxDigits);
   if ( bw1 < xmicros && absMaxLog<0)
   {
      // First case : bin width less than 0.001
      fExp = (Int_t)absMaxLog;
      if (fExp%3 == 1) fExp += TMath::Sign(2, fExp);
      if (fExp%3 == 2) fExp += TMath::Sign(1, fExp);
      if1     = fMaxDigits;
      if2     = fMaxDigits-2;
   }
   else
   {
      // Use x 10 n format. (only powers of 3 allowed)
      Float_t af = (absMax > 1) ? absMaxLog : TMath::Log10(absMax*0.0001);
      af += epsilon;
      Int_t clog = Int_t(af)+1;

      if (clog > fMaxDigits) {
         while (1) {
            fExp++;
            absMax    /= 10;
            if (fExp%3 == 0 && absMax <= TMath::Power(10,fMaxDigits-1)) break;
         }
      }
      else if (clog < -fMaxDigits) {
         Double_t rne   = 1/TMath::Power(10,fMaxDigits-2);
         while (1) {
            fExp--;
            absMax  *= 10;
            if (fExp%3 == 0 && absMax >= rne) break;
         }
      }

      Int_t na = 0;
      for (Int_t i=fMaxDigits-1; i>0; i--) {
         if (TMath::Abs(absMax) < TMath::Power(10,i)) na = fMaxDigits-i;
      }
      Double_t size =  TMath::Abs(fAtt->fMax - fAtt->fMin);
      Int_t ndyn = (Int_t)(size/bw1);
      while (ndyn) {
         if ( size/ndyn <= 0.999 && na < fMaxDigits-2) {
            na++;
            ndyn /= 10;
         }
         else break;
      }
      if2 = na;
      if1 = TMath::Max(clog+na,fMaxDigits)+1;
   }

   // compose text format
   if (TMath::Min(fAtt->fMin,fAtt->fMax) < 0)if1 = if1+1;
   if1 = TMath::Min(if1,32);

   // In some cases, if1 and if2 are too small....
   Double_t dwlabel = bw1*TMath::Power(10, -fExp);
   while (dwlabel < TMath::Power(10,-if2)) {
      if1++;
      if2++;
   }
   if (if1 > 14) if1=14;
   if (if2 > 14) if2=14;
   if (if2) sprintf(fFormat,"%%%d.%df",if1,if2);
   else     sprintf(fFormat,"%%%d.%df",if1+1,1);

   // get decimal number
   char chtemp[8];
   sprintf(chtemp,"%g",dwlabel);
   fDecimals = 0;
   char *dot = strchr(chtemp,'.');
   if (dot) fDecimals = chtemp + strlen(chtemp) -dot;
}


//______________________________________________________________________________
void TGLAxisPainter::RnrText(const char* txt, TGLVector3 pos, TGLFont &font) const
{
   // Render text at the given position. Offset depends of text aligment.

   glPushMatrix();
   glTranslatef(pos.X(), pos.Y(), pos.Z());

   Float_t llx, lly, llz, urx, ury, urz;
   font.BBox(txt, llx, lly, llz, urx, ury, urz);
   if (txt[0] == '-')
      urx += (urx-llx)/strlen(txt);

   Float_t x=0, y=0;

   switch (fAtt->fTextAlign)
   {
      case TGLFont::kCenterDown:
         x = -urx*0.5; y = -ury;
         break;
      case TGLFont::kCenterUp:
         x = -urx; y = 0;
         break;
      case TGLFont::kLeft:
         x = -urx; y =(lly -ury)*0.5;
         break;
      case TGLFont::kRight:
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

   //______________________________________________________________________________

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
      att.fAbsLabelFontSize = TGLFontManager::GetFontSize(fs, 8, 36);
      rnrCtx.RegisterFont(att.fAbsLabelFontSize, att.fLabelFontName.Data(), TGLFont::kPixmap, font);

      TGLUtil::Color(att.fLabelColor);
      glPushMatrix();
      TGLVector3 off = (att.fTMOff[0])*2.5; // tmp
      glTranslated (off.X(), off.Y(), off.Z());

      font.PreRender();
      TGLVector3 pos  = att.fDir*bl1;
      TGLVector3 step = att.fDir*bw1;
      SetTextFormat(bw1);
      Double_t lab0 = bl1*TMath::Power(10, -fExp);
      Double_t labStep = bw1*TMath::Power(10, -fExp);
      char chtemp[10];
      for (Int_t i=0; i<=bn1; i++)
      {
         FormAxisValue(lab0+i*labStep, &chtemp[0]);
         font.RenderBitmap(chtemp, pos.X(), pos.Y(), pos.Z(), att.fTextAlign);
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
      att.fAbsTitleFontSize = TGLFontManager::GetFontSize(fs, 12, 36);

      rnrCtx.RegisterFont(TGLFontManager::GetFontSize(fs, 12, 36),
                          att.fTitleFontName.Data(), TGLFont::kPixmap, font);
      TGLUtil::Color(att.fTitleColor);
      font.PreRender();
      TGLVector3 pos = att.fTitlePos;
      pos  += att.fTMOff[0]*2.5; //tmp

      TString title = att.fTitle;
      if (att.fTitleUnits.Length())
      {
         if (fExp)
            title += Form("[10^%d %s]", fExp, att.fTitleUnits.Data());
         else
            title += Form("[%s]", att.fTitleUnits.Data());
      }
      RnrText(title.Data(), pos, font);

      font.PostRender();
      rnrCtx.ReleaseFont(font);
   }

   //______________________________________________________________________________

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
