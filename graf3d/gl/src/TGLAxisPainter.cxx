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
#include "TPRegexp.h"

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

   fFontMode(TGLFont::kTexture),
   fDir(1, 0, 0),
   fTMNDim(1),
   fLabelPixelFontSize(14),
   fTitlePixelFontSize(14)
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
void TGLAxisPainter::FormAxisValue(Double_t  val, TString &s) const
{
   // Returns formatted text suitable for display of value.

   static char label[256];

   sprintf(label, &fFormat[0], val);
   s =  label;

   if (s == "-." || s == "-0")
   {
      s  = "0";
      return;
   }

   if (s.EndsWith("."))
      s += '0';

   Ssiz_t ld = s.Last('.');
   if (s.Length() - ld > fDecimals)
      s.Remove(ld + fDecimals);

   TPMERegexp zeroes("[-+]?0\\.0*$");
   zeroes.Substitute(s, "0");   
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
void TGLAxisPainter::RnrText( const char* txt, const TGLVector3 &pos, const TGLFont::ETextAlign_e align, const TGLFont &font) const
{
   // Render text at the given position. Offset depends of text aligment.

   glPushMatrix();

   glTranslatef(pos.X(), pos.Y(), pos.Z());
   Float_t llx, lly, llz, urx, ury, urz;
   font.BBox(txt, llx, lly, llz, urx, ury, urz);

   Float_t x=0, y=0;
   switch (align)
   {
      case TGLFont::kCenterUp:
         if (txt[0] == '-')
            urx += (urx-llx)/strlen(txt);
         x = -urx*0.5; y = -ury;
         break;
      case TGLFont::kCenterDown:
         if (txt[0] == '-')
            urx += (urx-llx)/strlen(txt);
         x = -urx*0.5; y = 0;
         break;
      case TGLFont::kRight:
         x = -urx; y =(lly -ury)*0.5;
         break;
      case TGLFont::kLeft:
         x = 0; y = -ury*0.5;
         break;
      default:
         break;
   };


   if (fFontMode == TGLFont::kPixmap || fFontMode ==  TGLFont::kBitmap)
   {
      glRasterPos2i(0, 0);
      glBitmap(0, 0, 0, 0, x, y, 0);
   }
   else
   {
      Double_t sc = fLabel3DFontSize/fLabelPixelFontSize;
      glScaled(sc, sc, 1);
      glTranslatef(x, y, 0);
   }

   font.Render(txt);
   glPopMatrix();
}

//______________________________________________________________________________
void TGLAxisPainter::SetLabelFont(TGLRnrCtx &rnrCtx, const char* fontName, Int_t fontSize, Double_t size3d)
{
   // Set label font derived from TAttAxis.

   fLabelPixelFontSize = TGLFontManager::GetFontSize(fontSize, 10, 128);
   fLabel3DFontSize = size3d;
   if (fLabelFont.GetMode() == TGLFont::kUndef)
   {
      rnrCtx.RegisterFont(fLabelPixelFontSize, fontName, fFontMode, fLabelFont);
   }
   else if (fLabelFont.GetSize() != fontSize|| fLabelFont.GetFile() != fAttAxis->GetLabelFont() || fLabelFont.GetMode() != fFontMode )
   {
      rnrCtx.ReleaseFont(fLabelFont);
      rnrCtx.RegisterFont(fLabelPixelFontSize, fontName, fFontMode, fLabelFont);
   }
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
   Double_t p = 0.;
   TString s;
   for (LabVec_t::const_iterator it = fLabVec.begin(); it != fLabVec.end(); ++it) {
      FormAxisValue((*it).second, s);
      p = (*it).first;
      RnrText(s.Data(), fDir*p, fLabelAlign, fLabelFont);
   }

   fLabelFont.PostRender();
   glPopMatrix();
}

//______________________________________________________________________________
void TGLAxisPainter::SetTitleFont(TGLRnrCtx &rnrCtx, const char* fontName, Int_t fontSize, Double_t size3d)
{
   // Set title font derived from TAttAxis.

   fTitlePixelFontSize = TGLFontManager::GetFontSize(fontSize, 10, 128);
   fTitle3DFontSize = size3d;

   if (fTitleFont.GetMode() == TGLFont::kUndef)
   {
      rnrCtx.RegisterFont(fontSize, fontName, fFontMode, fTitleFont);
   }
   else if (fTitleFont.GetSize() != fontSize|| fTitleFont.GetFile() != fAttAxis->GetTitleFont() || fTitleFont.GetMode() != fFontMode )
   {
      rnrCtx.ReleaseFont(fTitleFont);
      rnrCtx.RegisterFont(fTitlePixelFontSize, fontName, fFontMode, fTitleFont);
   }
}

//______________________________________________________________________________
void TGLAxisPainter::RnrTitle(const char* txt, TGLVector3 &pos , TGLFont::ETextAlign_e align) const
{
   // Draw title at given position.

   TGLUtil::Color(fAttAxis->GetTitleColor());
   const char* title = (fExp) ? Form("%s [10^%d]", fExp, txt) : txt;
   fTitleFont.PreRender();
   RnrText(title, pos, align, fTitleFont);
   fTitleFont.PostRender();
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
   Int_t nt =  fTMVec.size()-1;
   it++;
   for (Int_t t = 1; t < nt; ++t, ++it) {
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
   Double_t min = ax->GetXmin();
   Double_t max = ax->GetXmax();
   if (min == max)
   {
      Error("TGLAxisPainter::PaintAxis", "axis without range");
      return;
   }

   //______________________________________________________________________________
   // Fill lablels value-pos and tick-marks position-length.

   Int_t n1a = TMath::FloorNint(fAttAxis->GetNdivisions() / 100);
   Int_t n2a = fAttAxis->GetNdivisions() - n1a * 100;
   Int_t bn1, bn2;
   Double_t bw1, bw2; // primary , secondary bin width
   Double_t bl1, bh1, bl2, bh2; // bin low, high values

   // Read limits from users range
   THLimitsFinder::Optimize(min, max, n1a, bl1, bh1, bn1, bw1);
   THLimitsFinder::Optimize(bl1, bl1 + bw1, n2a, bl2, bh2, bn2, bw2);

   //______________________________________________________________________________

   // Get TM. First and last values are reserved for axis range
   //
   fTMVec.clear();
   fLabVec.clear();

   fTMVec.push_back(TM_t(min, -1));

   Double_t v1 = bl1;
   Double_t v2 = 0;
   for (Int_t t1 = 0; t1 <= bn1; t1++)
   {
      fTMVec.push_back(TM_t(v1, 0));
      fLabVec.push_back(Lab_t(v1, v1));
      v2 = v1 + bw2;
      for (Int_t t2 = 1; t2 < bn2; t2++)
      {
         if (v2 > max) break;
         fTMVec.push_back(TM_t(v2, 1));
         v2 += bw2;
      }
      v1 += bw1;
   }

   // complete low edges for 1.st order TM
   v2 = bl1 -bw2;
   while (v2 > min) {
      fTMVec.push_back(TM_t(v2, 1));
      v2 -= bw2;
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
   const char* labFontName = TGLFontManager::GetFontNameFromId(fAttAxis->GetLabelFont());
   const char* titleFontName = TGLFontManager::GetFontNameFromId(fAttAxis->GetTitleFont());

   if (fFontMode == TGLFont::kPolygon || fFontMode == TGLFont::kTexture)
   {
      // get sensible pixel resolution relative to projected axis length
      // in pixmap for this is given explicitly
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

      fLabelPixelFontSize = TMath::CeilNint(len*fAttAxis->GetLabelSize());
      fTitlePixelFontSize = TMath::CeilNint(len*fAttAxis->GetTitleSize());
   }

   SetLabelFont(rnrCtx, labFontName, fLabelPixelFontSize,   (max -min)*fAttAxis->GetLabelSize());
   SetTitleFont(rnrCtx, titleFontName, fTitlePixelFontSize, (max -min)*fAttAxis->GetTitleSize());

   //______________________________________________________________________________
   // Draw.

   glDisable(GL_LIGHTING);
   RnrLines();
   RnrLabels();

   if (ax->GetTitle())
      RnrTitle(ax->GetTitle(), fTitlePos, fLabelAlign);
}
