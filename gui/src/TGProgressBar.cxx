// @(#)root/gui:$Name:  $:$Id: TGProgressBar.cxx,v 1.1 2000/10/09 19:13:30 rdm Exp $
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
   fShowPos    = kFALSE;
   fPercent    = kTRUE;
   fNormGC     = norm;
   fFontStruct = font;
   fBarColorGC.SetForeground(barcolor);
}

//______________________________________________________________________________
void TGProgressBar::SetRange(Float_t min, Float_t max)
{
   // Set min and max of progress bar. Must be called before position is set.

   if (min >= max) {
      Error("SetRange", "max must be > min");
      return;
   }
   if (fPos > 0) {
      Error("SetRange", "must be called before position is incremented");
      return;
   }

   fMin = min;
   fMax = max;
}

//______________________________________________________________________________
void TGProgressBar::SetPosition(Float_t pos)
{
   // Set progress position between [min,max].

   if (pos < fMin) pos = fMin;
   if (pos > fMax) pos = fMax;
   fPos = pos;

   //fClient->NeedRedraw(this);
   fDrawBar = kTRUE;
   DoRedraw();
}

//______________________________________________________________________________
void TGProgressBar::Increment(Float_t inc)
{
   // Increment progress position.

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
{ return fgDefaultFontStruct; }

//______________________________________________________________________________
const TGGC &TGProgressBar::GetDefaultGC()
{ return fgDefaultGC; }


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

   if (fType == kSolidFill)
      gVirtualX->FillRectangle(fId, fBarColorGC(), fBorderWidth,
                               fBorderWidth, fPosPix - fBorderWidth, fBarWidth -
                               (fBorderWidth << 1));
   else {
      Int_t blocksize = 15;
      Int_t delta     = 2;
      Int_t pos       = fBorderWidth;
      while (pos < fPosPix) {
         if (pos + blocksize > fPosPix)
            blocksize = fPosPix-pos;
         gVirtualX->FillRectangle(fId, fBarColorGC(), pos,
                                  fBorderWidth, blocksize, fBarWidth -
                                  (fBorderWidth << 1));
         if (fDrawBar && fShowPos)
            gVirtualX->ClearArea(fId, pos+blocksize, fBorderWidth,
                                 delta, fBarWidth - (fBorderWidth << 1));

         pos += blocksize + delta;
      }
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

      if (fDrawBar && Int_t(x+twidth) > fPosPix)
         gVirtualX->ClearArea(fId, fPosPix, fBorderWidth,
                              fWidth - fPosPix- fBorderWidth,
                              fBarWidth - (fBorderWidth << 1));

      gVirtualX->DrawString(fId, fNormGC, x, y + max_ascent, buf, strlen(buf));
   }

   fDrawBar = kFALSE;
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
      Int_t blocksize = 15;
      Int_t delta     = 2;
      Int_t pos       = fBorderWidth;
      while (pos < fPosPix) {
         if (pos + blocksize > fPosPix)
            blocksize = fPosPix-pos;
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

