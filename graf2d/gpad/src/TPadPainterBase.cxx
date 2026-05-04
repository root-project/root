// @(#)root/gpad:$Id$
// Author:  Sergey Linev  17/04/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TPadPainterBase.h"
#include "TColor.h"

#include "TTF.h"
#include "TVirtualX.h"
#include "TMathBase.h"

/** \class TPadPainterBase
\ingroup gpad

Extends TVirtualPadPainter interface to simplify work with graphical attributes

Plus for now central place for TTF handling
*/

////////////////////////////////////////////////////////////////////////////////
/// Returns fill attributes after modification
/// Checks for special fill styles 4000 .. 4100

TAttFill TPadPainterBase::GetAttFillInternal(Bool_t with_transparency)
{
   Style_t style = GetAttFill().GetFillStyle();
   Color_t color = GetAttFill().GetFillColor();

   fFullyTransparent = (style == 4000) || (style == 0);
   if (fFullyTransparent) {
      style = 0;
   } else if ((style > 4000) && (style <= 4100)) {
      if ((style < 4100) && with_transparency)
         color = TColor::GetColorTransparent(color, (style - 4000) / 100.);
      style = 1001;
   }

   return { color, style };
}

////////////////////////////////////////////////////////////////////////////////
/// Returns text extend

void TPadPainterBase::GetTextExtent(Font_t font, Double_t size, UInt_t &w, UInt_t &h, const char *mess)
{
   Bool_t res = kFALSE;

   if (!HasTTFonts() && gVirtualX)
      res = gVirtualX->GetTextExtentA(font, size, w, h, mess);

   if (!res) {
      TTFhandle ttf;
      ttf.SetTextFont(font);
      ttf.SetTextSize(size);
      ttf.GetTextExtent(w, h, mess);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Returns text extend

void TPadPainterBase::GetTextExtent(Font_t font, Double_t size, UInt_t &w, UInt_t &h, const wchar_t *mess)
{
   Bool_t res = kFALSE;

   if (!HasTTFonts() && gVirtualX)
      res = gVirtualX->GetTextExtentA(font, size, w, h, mess);

   if (!res) {
      TTFhandle ttf;
      ttf.SetTextFont(font);
      ttf.SetTextSize(size);
      ttf.GetTextExtent(w, h, mess);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Returns text accent / descent

void TPadPainterBase::GetTextAscentDescent(Font_t font, Double_t size, UInt_t &a, UInt_t &d, const char *mess)
{
   Bool_t res = kFALSE;

   if (!HasTTFonts() && gVirtualX) {
      res = gVirtualX->GetFontAscentDescent(font, size, a, d, mess);
      if (res & !a) {
         UInt_t w = 0;
         gVirtualX->GetTextExtentA(font, size, w, a, mess);
      }
   }

   if (!res) {
      TTFhandle ttf;
      ttf.SetTextFont(font);
      ttf.SetTextSize(size);
      UInt_t w, h;
      ttf.GetTextExtent(w, h, mess);
      a = ttf.GetBox().yMax;
      d = TMath::Abs(ttf.GetBox().yMin);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Returns text accent / descent

void TPadPainterBase::GetTextAscentDescent(Font_t font, Double_t size, UInt_t &a, UInt_t &d, const wchar_t *mess)
{
   Bool_t res = kFALSE;

   // special use case for MacOS - directly use TTF
   if (!HasTTFonts() && !IsCocoa() && gVirtualX) {
      res = gVirtualX->GetFontAscentDescent(font, size, a, d, "");
      if (res & !a) {
         UInt_t w = 0;
         gVirtualX->GetTextExtentA(font, size, w, a, mess);
      }
   }

   if (!res) {
      TTFhandle ttf;
      ttf.SetTextFont(font);
      ttf.SetTextSize(size);
      UInt_t w, h;
      ttf.GetTextExtent(w, h, mess);
      a = ttf.GetBox().yMax;
      d = TMath::Abs(ttf.GetBox().yMin);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Returns text advance

UInt_t TPadPainterBase::GetTextAdvance(Font_t font, Double_t size, const char *mess, Bool_t kern)
{
   if (!HasTTFonts() && gVirtualX) {
      UInt_t a = 0, h;
      if (gVirtualX->GetTextExtentA(font, size, a, h, mess))
         return a;
   }

   TTFhandle ttf;
   ttf.SetTextFont(font);
   ttf.SetTextSize(size);
   ttf.SetKerning(kern);

   UInt_t a = 0;
   ttf.GetTextAdvance(a, mess);
   return a;
}
