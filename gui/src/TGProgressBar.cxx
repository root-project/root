// @(#)root/gui:$Name:$:$Id:$
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
// be used show progress of tasks taken more then a few seconds.        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGProgressBar.h"


ClassImp(TGProgressBar)
ClassImp(TGHProgressBar)
ClassImp(TGVProgressBar)

//______________________________________________________________________________
TGProgressBar::TGProgressBar(const TGWindow *p, UInt_t w, UInt_t h,
                             ULong_t back, GContext_t barcolor, GContext_t norm,
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
   fBarColorGC = barcolor;
   fNormGC     = norm;
   fFontStruct = font;
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

   fClient->NeedRedraw(this);
}

//______________________________________________________________________________
void TGProgressBar::Increment(Float_t inc)
{
   // Increment progress position.

   fPos += inc;
   if (fPos > fMax) fPos = fMax;

   fClient->NeedRedraw(this);
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
void TGProgressBar::ShowPosition(Bool_t set)
{
   // Show postion text.

   fShowPos = set;

   fClient->NeedRedraw(this);
}

//______________________________________________________________________________
void TGProgressBar::SetPercent(Bool_t set)
{
   // Position is specified in percent in [min,max].

   fPercent = set;

   fClient->NeedRedraw(this);
}

//______________________________________________________________________________
FontStruct_t TGProgressBar::GetDefaultFontStruct()
{ return fgDefaultFontStruct; }

//______________________________________________________________________________
const TGGC &TGProgressBar::GetDefaultGC()
{ return fgDefaultGC; }

//______________________________________________________________________________
const TGGC &TGProgressBar::GetDefaultBarColorGC()
{ return fgDefaultBarColorGC; }


//______________________________________________________________________________
void TGHProgressBar::DoRedraw()
{
   // Draw horizontal progress bar.

   // calls TGProgressBar::DrawBorder()
   TGFrame::DoRedraw();

   fPos = 1;

   fPosPix = Int_t(((Float_t)fWidth - (fBorderWidth << 1)) *
             (fPos - fMin) / (fMax - fMin) +
             fBorderWidth);

   if (fType == kSolidFill)
      gVirtualX->FillRectangle(fId, fBarColorGC, fBorderWidth,
                               fBorderWidth, fPosPix - fBorderWidth, fBarWidth -
                               (fBorderWidth << 1));
   else {


   }
}


//______________________________________________________________________________
void TGVProgressBar::DoRedraw()
{
   // Draw vertical progress bar.

   // calls TGProgressBar::DrawBorder()
   TGFrame::DoRedraw();

   fPos = 1;

   fPosPix = Int_t(((Float_t)fHeight - (fBorderWidth << 1)) *
             (fPos - fMin) / (fMax - fMin) +
             fBorderWidth);

   if (fType == kSolidFill)
      gVirtualX->FillRectangle(fId, fBarColorGC, fBorderWidth,
                               fHeight - fPosPix, fBarWidth - (fBorderWidth << 1),
                               fPosPix - fBorderWidth);
   else {

   }
}

