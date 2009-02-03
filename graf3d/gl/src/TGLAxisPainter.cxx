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

#include "TAttAxis.h"
#include "TAxis.h"
#include "THLimitsFinder.h"

#include "TMath.h"


//______________________________________________________________________________
//
// Utility class to paint axis in GL.


ClassImp(TGLAxisPainter);

//______________________________________________________________________________
TGLAxisPainter::TGLAxisPainter():
      fExp(0),
      fMaxDigits(5),
      fDecimals(0),

      fAttAxis(0),

      fDir(1, 0, 0),
      fTMNDim(1),
      fUseRelativeFontSize(kTRUE),
      fAbsoluteLabelFontSize(14),
      fAbsoluteTitleFontSize(14)
{
   // Constructor.
}


//______________________________________________________________________________
TGLAxisPainter::~TGLAxisPainter()
{
   // Destructor.

}

//______________________________________________________________________________
void TGLAxisPainter::LabelsLimits(const char *label, Int_t &first, Int_t &last) const
{
   // Find first and last character of a label.

   last = strlen(label) - 1;
   for (Int_t i = 0; i <= last; i++) {
      if (strchr("1234567890-+.", label[i])) {
         first = i;
         return;
      }
   }
   Error("LabelsLimits", "attempt to draw a blank label");
}

//______________________________________________________________________________
void TGLAxisPainter::FormAxisValue(Float_t wlabel, char* label) const
{
   // Returns formatted text suitable for display of value.

   sprintf(label, &fFormat[0], wlabel);
   Int_t first, last;
   LabelsLimits(label, first, last);

   char chtemp[256];
   if (label[first] == '.') { //check if '.' is preceeded by a digit
      strcpy(chtemp, "0");
      strcat(chtemp, &label[first]);
      strcpy(label, chtemp);
      first = 1;
      last = strlen(label);
   }
   if (label[first] == '-' && label[first+1] == '.') {
      strcpy(chtemp, "-0");
      strcat(chtemp, &label[first+1]);
      strcpy(label, chtemp);
      first = 1;
      last = strlen(label);
   }

   //  We eliminate the non significant 0 after '.'
   if (fDecimals) {
      char *adot = strchr(label, '.');
      if (adot) adot[fDecimals] = 0;
   } else {
      while (label[last] == '0') {
         label[last] = 0;
         last--;
      }
   }
   // We eliminate the dot, unless dot is forced.
   if (label[last] == '.') {
      label[last] = 0;
      last--;
   }

   //  Make sure the label is not "-0"
   if (last - first == 1 && label[first] == '-' && label[last]  == '0') {
      strcpy(label, "0");
      label[last] = 0;
   }

   // Remove white space
   Int_t cnt;
   for (cnt=0; cnt<last; cnt++)
      if (label[cnt] != ' ') break;

   strcpy(label, &label[cnt]);
}

//______________________________________________________________________________
void TGLAxisPainter::SetTextFormat(Double_t min, Double_t max, Double_t bw1)
{
   // Construct print format from given primary bin width.

   Double_t absMax = TMath::Max(TMath::Abs(min), TMath::Abs(max));
   Double_t epsilon = 1e-5;
   Double_t absMaxLog = TMath::Log10(absMax) + epsilon;

   fExp   = 0;
   Int_t if1, if2;
   Double_t xmicros = TMath::Power(10, -fMaxDigits);
   if (bw1 < xmicros && absMaxLog < 0) {
      // First case : bin width less than 0.001
      fExp = (Int_t)absMaxLog;
      if (fExp % 3 == 1) fExp += TMath::Sign(2, fExp);
      if (fExp % 3 == 2) fExp += TMath::Sign(1, fExp);
      if1     = fMaxDigits;
      if2     = fMaxDigits - 2;
   } else {
      // Use x 10 n format. (only powers of 3 allowed)
      Float_t af = (absMax > 1) ? absMaxLog : TMath::Log10(absMax * 0.0001);
      af += epsilon;
      Int_t clog = Int_t(af) + 1;

      if (clog > fMaxDigits) {
         while (1) {
            fExp++;
            absMax    /= 10;
            if (fExp % 3 == 0 && absMax <= TMath::Power(10, fMaxDigits - 1)) break;
         }
      } else if (clog < -fMaxDigits) {
         Double_t rne   = 1 / TMath::Power(10, fMaxDigits - 2);
         while (1) {
            fExp--;
            absMax  *= 10;
            if (fExp % 3 == 0 && absMax >= rne) break;
         }
      }

      Int_t na = 0;
      for (Int_t i = fMaxDigits - 1; i > 0; i--) {
         if (TMath::Abs(absMax) < TMath::Power(10, i)) na = fMaxDigits - i;
      }
      Double_t size =  TMath::Abs(max - min);
      Int_t ndyn = (Int_t)(size / bw1);
      while (ndyn) {
         if (size / ndyn <= 0.999 && na < fMaxDigits - 2) {
            na++;
            ndyn /= 10;
         } else break;
      }
      if2 = na;
      if1 = TMath::Max(clog + na, fMaxDigits) + 1;
   }

   // compose text format
   if (TMath::Min(min, max) < 0)if1 = if1 + 1;
   if1 = TMath::Min(if1, 32);

   // In some cases, if1 and if2 are too small....
   Double_t dwlabel = bw1 * TMath::Power(10, -fExp);
   while (dwlabel < TMath::Power(10, -if2)) {
      if1++;
      if2++;
   }
   if (if1 > 14) if1 = 14;
   if (if2 > 14) if2 = 14;
   if (if2) sprintf(fFormat, "%%%d.%df", if1, if2);
   else     sprintf(fFormat, "%%%d.%df", if1 + 1, 1);

   // get decimal number
   char chtemp[8];
   sprintf(chtemp, "%g", dwlabel);
   fDecimals = 0;
   char *dot = strchr(chtemp, '.');
   if (dot) fDecimals = chtemp + strlen(chtemp) - dot;
}

/**************************************************************************/
/**************************************************************************/
/**************************************************************************/
/**************************************************************************/

//
// Utility functions.

//______________________________________________________________________________
void TGLAxisPainter::SetLabelFont(TGLRnrCtx &rnrCtx, Double_t refLength)
{
   // Set label font derived from TAttAxis.

   Float_t len = (refLength < 0) ? (fTMVec.back().first - fTMVec.front().first) : refLength;
   Int_t fontSize = fUseRelativeFontSize ? Int_t(fAttAxis->GetLabelSize() * len) : fAbsoluteLabelFontSize;
   fontSize = TGLFontManager::GetFontSize(fontSize, 8, 36);
   const char* fontName = TGLFontManager::GetFontNameFromId(fAttAxis->GetLabelFont());

   if (fLabelFont.GetMode() == TGLFont::kUndef) {
      rnrCtx.RegisterFont(fontSize, fontName, TGLFont::kPixmap, fLabelFont);
   } else if (fLabelFont.GetSize() != fontSize) {
      rnrCtx.ReleaseFont(fLabelFont);
      rnrCtx.RegisterFont(fontSize, fontName, TGLFont::kPixmap, fLabelFont);
   }

   fAbsoluteLabelFontSize = fontSize;
}

//______________________________________________________________________________
void TGLAxisPainter::RnrLabels() const
{
   // Render label reading prepared list ov value-pos pairs.

   TGLUtil::Color(fAttAxis->GetLabelColor());

   glPushMatrix();

   Float_t off = fAttAxis->GetLabelOffset() +  fAttAxis->GetTickLength();
   TGLVector3 offVec = fTMOff[0] * off;
   glTranslated(offVec.X(), offVec.Y(), offVec.Z());

   fLabelFont.PreRender();
   char ctmp[10];
   for (LabVec_t::const_iterator it = fLabVec.begin(); it != fLabVec.end(); ++it) {
      FormAxisValue((*it).second, &ctmp[0]);
      fLabelFont.RenderBitmap(&ctmp[0],
            fDir.X()*(*it).first, fDir.Y()*(*it).first, fDir.Z()*(*it).first,
            fLabelAlign);
   }

   fLabelFont.PostRender();
   glPopMatrix();
}

//______________________________________________________________________________
void TGLAxisPainter::SetTitleFont(TGLRnrCtx &rnrCtx, Double_t refLength)
{
   // Set title font derived from TAttAxis.

   Float_t len = (refLength < 0) ? (fTMVec.back().first - fTMVec.front().first) : refLength;
   Int_t fontSize = fUseRelativeFontSize ? Int_t(fAttAxis->GetTitleSize() * len) : fAbsoluteTitleFontSize;
   fontSize = TGLFontManager::GetFontSize(fontSize, 8, 36);
   const char* fontName = TGLFontManager::GetFontNameFromId(fAttAxis->GetTitleFont());

   if (fTitleFont.GetMode() == TGLFont::kUndef) {
      rnrCtx.RegisterFont(fontSize, fontName, TGLFont::kPixmap, fTitleFont);
   } else if (fTitleFont.GetSize() != fontSize || fTitleFont.GetFile() != fAttAxis->GetTitleFont()) {
      rnrCtx.ReleaseFont(fTitleFont);
      rnrCtx.RegisterFont(fontSize, fontName, TGLFont::kPixmap, fTitleFont);
   }

   fAbsoluteTitleFontSize = fontSize;
}

//______________________________________________________________________________
void TGLAxisPainter::RnrTitle(const char* txt, Float_t pos, TGLFont::ETextAlign_e align) const
{
   // Draw title at given position.

   if (txt)
   {
      TGLUtil::Color(fAttAxis->GetTitleColor());
      const char* title = (fExp) ? Form("%s [10^%d]", fExp, txt) : txt;
      fTitleFont.PreRender();
      fTitleFont.RenderBitmap(title, pos*fDir.X(), pos*fDir.Y(), pos*fDir.Z(), align);
      fTitleFont.PostRender();
   }
}

//______________________________________________________________________________
void TGLAxisPainter::RnrLines() const
{
   // Render axis main line and tickmarks.

   TGLUtil::Color(fAttAxis->GetAxisColor());
   glBegin(GL_LINES);

   // Main line.
   //
   Float_t min = fTMVec.front().first;
   Float_t max = fTMVec.back().first;
   TGLVector3 start = fDir * min;
   TGLVector3 end   = fDir * max;
   glVertex3dv(start.Arr());
   glVertex3dv(end.Arr());

   // Tick-marks.
   // Support three possible directions and two orders.
   //
   Float_t tmsOrderFirst  = fAttAxis->GetTickLength();
   Float_t tmsOrderSecond = tmsOrderFirst * 0.5;
   TGLVector3 pos;
   TMVec_t::const_iterator it = fTMVec.begin();
   Int_t nt =  fTMVec.size();
   for (Int_t t = 1; t < nt; t++) {
      it++;
      pos = fDir * ((*it).first);
      for (Int_t dim = 0; dim < fTMNDim; dim++) {
         glVertex3dv(pos.Arr());
         if ((*it).second)
            glVertex3dv((pos + fTMOff[dim]*tmsOrderSecond).Arr());
         else
            glVertex3dv((pos + fTMOff[dim]*tmsOrderFirst).Arr());
      }
   }
   glEnd();
}

//______________________________________________________________________________
void TGLAxisPainter::PaintAxis(TGLRnrCtx &rnrCtx, TAxis* ax)
{
   // GL render TAxis.

   fAttAxis = ax;

   //______________________________________________________________________________
   // Fill lablels value-pos and tick-marks position-length.

   Int_t n1a = TMath::FloorNint(fAttAxis->GetNdivisions() / 100);
   Int_t n2a = fAttAxis->GetNdivisions() - n1a * 100;
   Int_t bn1, bn2;
   Double_t bw1, bw2; // primary , secondary bin width
   Double_t bl1, bh1, bl2, bh2; // bin low, high values

   // Read limits from users range
   Double_t min = ax->GetBinLowEdge(ax->GetFirst());
   Double_t max = ax->GetBinUpEdge(ax->GetLast());
   THLimitsFinder::Optimize(min, max, n1a, bl1, bh1, bn1, bw1);
   THLimitsFinder::Optimize(bl1, bl1 + bw1, n2a, bl2, bh2, bn2, bw2);


   //______________________________________________________________________________

   // Get TM. First and last values are reserved for axis range
   //
   fTMVec.clear();
   fTMVec.push_back(TM_t(min, -1));

   Double_t tv1 = bl1;
   Double_t tv2 = 0;
   for (Int_t t1 = 0; t1 < bn1; t1++) {
      fTMVec.push_back(TM_t(tv1, 0));
      tv2 = tv1 + bw2;
      for (Int_t t2 = 0; t2 <= bn2; t2++) {
         fTMVec.push_back(TM_t(tv2, 1));
         tv2 += bw2;
      }
      tv1 += bw1;
   }
   // complete last TM
   fTMVec.push_back(TM_t(tv1, 0));
   // complete up edges for 1.st order TM
   Int_t nc = Int_t((max - bh1) / bw2);
   tv2 = bh1;
   for (Int_t t2 = 0; t2 <= nc; t2++)
   {
      fTMVec.push_back(TM_t(tv2, 1));
      tv2 += bw2;
   }
   // complete low edges for 1.st order TM
   nc = Int_t((bl1 - min) / bw2);
   tv2 = bl1;
   for (Int_t t2 = 0; t2 <= nc; t2++) {
      fTMVec.push_back(TM_t(tv2, 1));
      tv2 -= bw2;
   }

   fTMVec.push_back(TM_t(max, -1));

   //______________________________________________________________________________
   // Get labels. In this case trivial one-one mapping.

   Double_t p = bl1;
   fLabVec.clear();
   SetTextFormat(min, max, bw1);
   for (Int_t i = 0; i <= bn1; i++) {
      fLabVec.push_back(Lab_t(p, p));
      p += bw1;
   }

   //______________________________________________________________________________
   // Set font.

   // First projected axis length needed if use realtive font size.
   Double_t len = 0;
   GLdouble mm[16];
   GLdouble pm[16];
   GLint    vp[4];
   glGetDoublev(GL_MODELVIEW_MATRIX,  mm);
   glGetDoublev(GL_PROJECTION_MATRIX, pm);
   glGetIntegerv(GL_VIEWPORT, vp);

   GLdouble dn[3];
   GLdouble up[3];
   gluProject(fDir.X()*min, fDir.Y()*min, fDir.Z()*min, mm, pm, vp, &dn[0], &dn[1], &dn[2]);
   gluProject(fDir.X()*max, fDir.Y()*max, fDir.Z()*max, mm, pm, vp, &up[0], &up[1], &up[2]);
   len = TMath::Sqrt((up[0] - dn[0]) * (up[0] - dn[0])
                     + (up[1] - dn[1]) * (up[1] - dn[1])
                     + (up[2] - dn[2]) * (up[2] - dn[2]));

   SetLabelFont(rnrCtx, len);
   SetTitleFont(rnrCtx, len);

   //______________________________________________________________________________
   // Draw.

   RnrLines();
   RnrLabels();
}
