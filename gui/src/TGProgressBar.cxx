// @(#)root/gui:$Name:  $:$Id: TGProgressBar.cxx,v 1.7 2003/10/19 14:15:08 rdm Exp $
// Author: Fons Rademakers   10/10/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGProgressBar, TGHProgressBar and TGVProgressBar                     //
//                                                                      //
// The classes in this file implement progress bars. Progress bars can  //
// be used to show progress of tasks taking more then a few seconds.    //
// TGProgressBar is an abstract base class, use either TGHProgressBar   //
// or TGVProgressBar. TGHProgressBar can in addition show the position  //
// as text in the bar.                                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGProgressBar.h"
#include "TGResourcePool.h"
#include "Riostream.h"


const TGFont *TGProgressBar::fgDefaultFont = 0;
TGGC         *TGProgressBar::fgDefaultGC = 0;


ClassImp(TGProgressBar)
ClassImp(TGHProgressBar)
ClassImp(TGVProgressBar)

//______________________________________________________________________________
TGProgressBar::TGProgressBar(const TGWindow *p, UInt_t w, UInt_t h,
                             ULong_t back, ULong_t barcolor, GContext_t norm,
                             FontStruct_t font, UInt_t options) :
   TGFrame(p, w, h, options | kOwnBackground, back)
{
   // Create progress bar.

   fMin        = 0;
   fMax        = 100;
   fPos        = 0;
   fPosPix     = 0;
   fType       = kSolidFill;
   fBarType    = kStandard;
   fShowPos    = kFALSE;
   fPercent    = kTRUE;
   fNormGC     = norm;
   fFontStruct = font;
   fBarColorGC.SetForeground(barcolor);
}

//______________________________________________________________________________
void TGProgressBar::SetRange(Float_t min, Float_t max)
{
   // Set min and max of progress bar.

   if (min >= max) {
      Error("SetRange", "max must be > min");
      return;
   }

   Bool_t draw = kFALSE;
   if (fPos > fMin) {
      // already in progress... rescale
      if (fPos < min) fPos = min;
      if (fPos > max) fPos = max;
      draw = kTRUE;
   } else
      fPos = min;

   fMin = min;
   fMax = max;

   if (draw)
      DoRedraw();
}

//______________________________________________________________________________
void TGProgressBar::SetPosition(Float_t pos)
{
   // Set progress position between [min,max].

   if (pos < fMin) pos = fMin;
   if (pos > fMax) pos = fMax;

   if (fPos == pos)
      return;

   fPos = pos;

   //fClient->NeedRedraw(this);
   fDrawBar = kTRUE;
   DoRedraw();
}

//______________________________________________________________________________
void TGProgressBar::Increment(Float_t inc)
{
   // Increment progress position.

   if (fPos == fMax)
      return;

   fPos += inc;
   if (fPos > fMax) fPos = fMax;

   //fClient->NeedRedraw(this);
   fDrawBar = kTRUE;
   DoRedraw();
}

//______________________________________________________________________________
void TGProgressBar::Reset()
{
   // Reset progress bar (i.e. set pos to 0).

   fPos = 0;

   fClient->NeedRedraw(this);
}

//______________________________________________________________________________
void TGProgressBar::SetFillType(EFillType type)
{
   // Set fill type.

   fType = type;

   fClient->NeedRedraw(this);
}

//______________________________________________________________________________
void TGProgressBar::SetBarColor(ULong_t color)
{
   // Set progress bar color.

   fBarColorGC.SetForeground(color);

   fClient->NeedRedraw(this);
}

//______________________________________________________________________________
void TGProgressBar::SetBarColor(const char *color)
{
   // Set progress bar color.

   ULong_t ic;
   fClient->GetColorByName(color, ic);

   fBarColorGC.SetForeground(ic);

   fClient->NeedRedraw(this);
}

//______________________________________________________________________________
FontStruct_t TGProgressBar::GetDefaultFontStruct()
{
   if (!fgDefaultFont)
      fgDefaultFont = gClient->GetResourcePool()->GetDefaultFont();
   return fgDefaultFont->GetFontStruct();
}

//______________________________________________________________________________
const TGGC &TGProgressBar::GetDefaultGC()
{
   if (!fgDefaultGC)
      fgDefaultGC = new TGGC(*gClient->GetResourcePool()->GetFrameGC());
   return *fgDefaultGC;
}


//______________________________________________________________________________
TGHProgressBar::TGHProgressBar(const TGWindow *p, EBarType type, UInt_t w)
   : TGProgressBar(p, w, type == kStandard ? kProgressBarStandardWidth :
                   kProgressBarTextWidth, type == kStandard ? GetDefaultFrameBackground() :
                   fgWhitePixel, fgDefaultSelectedBackground, GetDefaultGC()(),
                   GetDefaultFontStruct(),
                   type == kStandard ? kSunkenFrame : kDoubleBorder | kSunkenFrame)
{
   // Simple constructor allow you to create either a standard progress
   // bar, or a more fancy progress bar (fancy means: double sized border,
   // white background and a bit wider to allow for text to be printed
   // in the bar.

   fBarType  = type;
   fBarWidth = (type == kStandard) ? kProgressBarStandardWidth : kProgressBarTextWidth;
}

//______________________________________________________________________________
void TGHProgressBar::ShowPosition(Bool_t set, Bool_t percent, const char *format)
{
   // Show postion text, either in percent or formatted according format.

   fShowPos = set;
   fPercent = percent;
   fFormat  = format;

   fClient->NeedRedraw(this);
}

//______________________________________________________________________________
void TGHProgressBar::DoRedraw()
{
   // Draw horizontal progress bar.

   if (!fDrawBar) {
      // calls TGProgressBar::DrawBorder()
      TGFrame::DoRedraw();
   }

   fPosPix = Int_t(((Float_t)fWidth - (fBorderWidth << 1)) *
             (fPos - fMin) / (fMax - fMin) +
             fBorderWidth);

   Int_t pospix = fPosPix;

   if (fType == kSolidFill)
      gVirtualX->FillRectangle(fId, fBarColorGC(), fBorderWidth,
                               fBorderWidth, fPosPix - fBorderWidth, fBarWidth -
                               (fBorderWidth << 1));
   else {
      Int_t blocksize = kBlockSize;
      Int_t delta     = kBlockSpace;
      Int_t pos       = fBorderWidth;
      while (pos < fPosPix) {
         if (pos + blocksize > Int_t(fWidth)-fBorderWidth)
            blocksize = fWidth-fBorderWidth-pos;
         gVirtualX->FillRectangle(fId, fBarColorGC(), pos,
                                  fBorderWidth, blocksize, fBarWidth -
                                  (fBorderWidth << 1));
         if (fDrawBar && fShowPos)
            gVirtualX->ClearArea(fId, pos+blocksize, fBorderWidth,
                                 delta, fBarWidth - (fBorderWidth << 1));

         pos += blocksize + delta;
      }
      pospix = pos - delta;
   }

   if (fShowPos) {
      char buf[256];
      if (fPercent)
         sprintf(buf, "%d%%", Int_t((fPos-fMin)/(fMax-fMin)*100.));
      else
         sprintf(buf, fFormat.Data(), fPos);

      Int_t x, y, max_ascent, max_descent;
      UInt_t twidth  = gVirtualX->TextWidth(fFontStruct, buf, strlen(buf));
      gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);
      UInt_t theight = max_ascent + max_descent;

      x = (fWidth - twidth) >> 1;
      y = (fHeight - theight) >> 1;

      if (fDrawBar && fPosPix < Int_t(x+twidth))
         gVirtualX->ClearArea(fId, pospix, fBorderWidth,
                              fWidth - pospix - fBorderWidth,
                              fBarWidth - (fBorderWidth << 1));

      gVirtualX->DrawString(fId, fNormGC, x, y + max_ascent, buf, strlen(buf));
   }

   fDrawBar = kFALSE;
}


//______________________________________________________________________________
TGVProgressBar::TGVProgressBar(const TGWindow *p, EBarType type, UInt_t h)
   : TGProgressBar(p, type == kStandard ? kProgressBarStandardWidth :
                   kProgressBarTextWidth, h, type == kStandard ? GetDefaultFrameBackground() :
                   fgWhitePixel, fgDefaultSelectedBackground, GetDefaultGC()(),
                   GetDefaultFontStruct(),
                   type == kStandard ? kSunkenFrame : kDoubleBorder | kSunkenFrame)
{
   // Simple constructor allow you to create either a standard progress
   // bar, or a more fancy progress bar (fancy means: double sized border,
   // white background and a bit wider to allow for text to be printed
   // in the bar.

   fBarType  = type;
   fBarWidth = (type == kStandard) ? kProgressBarStandardWidth : kProgressBarTextWidth;
}

//______________________________________________________________________________
void TGVProgressBar::DoRedraw()
{
   // Draw vertical progress bar.

   if (!fDrawBar) {
      // calls TGProgressBar::DrawBorder()
      TGFrame::DoRedraw();
   }

   fPosPix = Int_t(((Float_t)fHeight - (fBorderWidth << 1)) *
             (fPos - fMin) / (fMax - fMin) +
             fBorderWidth);

   if (fType == kSolidFill)
      gVirtualX->FillRectangle(fId, fBarColorGC(), fBorderWidth,
                               fHeight - fPosPix, fBarWidth - (fBorderWidth << 1),
                               fPosPix - fBorderWidth);
   else {
      Int_t blocksize = kBlockSize;
      Int_t delta     = kBlockSpace;
      Int_t pos       = fBorderWidth;
      while (pos < fPosPix) {
         if (pos + blocksize > Int_t(fHeight)-fBorderWidth)
            blocksize = fHeight-fBorderWidth-pos;
         gVirtualX->FillRectangle(fId, fBarColorGC(), fBorderWidth,
                                  fHeight - pos - blocksize, fBarWidth - (fBorderWidth << 1),
                                  blocksize);
         pos += blocksize + delta;
      }
   }

   if (fShowPos) {
      // not text shown for vertical progress bars
   }

   fDrawBar = kFALSE;
}

//______________________________________________________________________________
void TGProgressBar::SavePrimitive(ofstream &out, Option_t *)
{
   // Save progress bar parameters as a C++ statement(s) on output stream out

   if (fMin != 0 && fMax != 100)
      out << "   " << GetName() << "->SetRange(" << fMin << "," << fMax
      << ");" << endl;

   out <<"   "<< GetName() <<"->SetPosition("<< fPos <<");"<< endl;

}

//______________________________________________________________________________
void TGVProgressBar::SavePrimitive(ofstream &out, Option_t *option)
{
   // Save a vertical progress bar as a C++ statement(s) on output stream out


   out << "   TGVProgressBar *";
   out << GetName() << " = new TGVProgressBar(" << fParent->GetName();

   if (GetBarType()) {
       out << ",TGProgressBar::kFancy,";
   } else {
     out << ",TGProgressBar::kStandard,";
   }

   out << GetHeight() <<");" << endl;

   if (GetFillType() == kBlockFill)
      out << "   " << GetName() <<"->SetFillType(TGProgressBar::kBlockFill);"<< endl;

   out << "   " << GetName() << "->SetBarColor(" << fBarColorGC.GetForeground()
       << ");"  << endl;
   TGProgressBar::SavePrimitive(out, option);
}

//______________________________________________________________________________
void TGHProgressBar::SavePrimitive(ofstream &out, Option_t *option)
{
    // Save a vertical progress bar as a C++ statement(s) on output stream out

   char quote = '"';

   out<<"   TGHProgressBar *";
   out << GetName() <<" = new TGHProgressBar("<< fParent->GetName();

   if (GetBarType()) {
       out << ",TGProgressBar::kFancy,";
   } else {
     out << ",TGProgressBar::kStandard,";
   }

   out << GetWidth() << ");" << endl;

   if (GetFillType() == kBlockFill)
      out << "   " << GetName() <<"->SetFillType(TGProgressBar::kBlockFill);"<< endl;

   if (GetShowPos()) {
       out << "   " << GetName() <<"->ShowPosition(kTRUE,";
       if (UsePercent()) {
           out << "kTRUE,";
       } else {
         out << "kFALSE,";
       }
       out << quote << GetFormat() << quote << ");"<< endl;

   } else if (UsePercent() && !GetFillType()) {
      out << "   " << GetName() <<"->ShowPosition();" << endl;
   }
   out << "   " << GetName() <<"->SetBarColor(" << fBarColorGC.GetForeground()
       << ");"  << endl;
   TGProgressBar::SavePrimitive(out, option);

}
