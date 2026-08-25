// @(#)root/graf:$Id$
// Author: Olivier Couet     01/10/2002
// Author: Sergey  Linev     29/04/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_GLYPH_H
#include "TROOT.h"
#include "TTF.h"
#include "TSystem.h"
#include "TEnv.h"
#include "TMath.h"
#include "TError.h"


/** \class TTF
\ingroup BasicGraphics

Interface to the freetype 2 library.
Implements old static API.
Unitl ROOT7 just redirects to static TTFhandle instance,
then only TTFhandle class will remains
*/

thread_local TTF gCleanupTTF; // Allows to call "Cleanup" at the end of the session
thread_local std::unique_ptr<TTFhandle> fgHandle; // static handle, destroyed automatically

////////////////////////////////////////////////////////////////////////////////
/// Cleanup TTF environment.

TTF::~TTF() {}

////////////////////////////////////////////////////////////////////////////////
/// Init TTF environment.

void TTF::Init()
{
   if (!fgHandle) {
      fgHandle = std::make_unique<TTFhandle>();
      fgHandle->SetTextFont(62);
   }
}

////////////////////////////////////////////////////////////////////////////////

Bool_t TTF::GetHinting()
{
   return fgHandle ? fgHandle->GetHinting() : kFALSE;
}

////////////////////////////////////////////////////////////////////////////////

Bool_t TTF::GetKerning()
{
   return fgHandle ? fgHandle->GetKerning() : kFALSE;
}

////////////////////////////////////////////////////////////////////////////////

Bool_t TTF::GetSmoothing()
{
   return fgHandle ? fgHandle->GetSmoothing() : kTRUE;
}

////////////////////////////////////////////////////////////////////////////////

Bool_t TTF::IsInitialized()
{
   return fgHandle.get() != nullptr;
}

////////////////////////////////////////////////////////////////////////////////

Int_t TTF::GetWidth()
{
   return fgHandle ? fgHandle->GetWidth() : 0;
}

////////////////////////////////////////////////////////////////////////////////

Int_t TTF::GetAscent()
{
   return fgHandle ? fgHandle->GetAscent() : 0;
}

////////////////////////////////////////////////////////////////////////////////

Int_t  TTF::GetNumGlyphs()
{
   return fgHandle ? fgHandle->GetNumGlyphs() : 0;
}

////////////////////////////////////////////////////////////////////////////////

FT_Matrix *TTF::GetRotMatrix()
{
   static FT_Matrix m;
   if (fgHandle && (fgHandle->fRotationXX || fgHandle->fRotationXX)) {
      m.xx = m.yy = fgHandle->fRotationXX;
      m.xy = fgHandle->fRotationXY;
      m.yx = -fgHandle->fRotationXY;
      return &m;
   }

   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////

Int_t  TTF::GetTrailingBlanksWidth()
{
   return fgHandle ? fgHandle->GetTrailingBlanksWidth() : 0;
}

////////////////////////////////////////////////////////////////////////////////

const FT_BBox &TTF::GetBox()
{
   static FT_BBox bbox;
   if (fgHandle) {
      bbox.xMin = fgHandle->GetBoxXMin();
      bbox.yMin = fgHandle->GetBoxYMin();
      bbox.xMax = fgHandle->GetBoxXMax();
      bbox.yMax = fgHandle->GetBoxYMax();
   }
   return bbox;
}

////////////////////////////////////////////////////////////////////////////////

TTF::TTGlyph *TTF::GetGlyphs()
{
   static std::vector<TTF::TTGlyph> vect;

   if (fgHandle)
      fgHandle->FillTTFGlypths(&vect);

   return vect.data();
}

////////////////////////////////////////////////////////////////////////////////
/// Map char to unicode. Returns 0 in case no mapping exists.

Short_t TTF::CharToUnicode(UInt_t code)
{
   Init();
   return fgHandle->CharToUnicode(code);
}

////////////////////////////////////////////////////////////////////////////////
/// Set the rotation matrix used to rotate the font outlines.

void TTF::SetRotationMatrix(Float_t angle)
{
   Init();
   fgHandle->SetRotationMatrix(angle);
}

////////////////////////////////////////////////////////////////////////////////
/// Set hinting flag.

void TTF::SetHinting(Bool_t state)
{
   Init();
   fgHandle->SetHinting(state);
}

////////////////////////////////////////////////////////////////////////////////
/// Set kerning flag.

void TTF::SetKerning(Bool_t state)
{
   Init();
   fgHandle->SetKerning(state);
}

////////////////////////////////////////////////////////////////////////////////
/// Set smoothing (anti-aliasing) flag.

void TTF::SetSmoothing(Bool_t state)
{
   Init();
   fgHandle->SetSmoothing(state);
}

////////////////////////////////////////////////////////////////////////////////
/// Set text font to specified name.
///  - font       : font name
///  - italic     : the fonts should be slanted. Used for symbol font.

Int_t TTF::SetTextFont(const char *fontname, Int_t italic)
{
   Init();
   return fgHandle->SetTextFont(fontname, italic);
}

////////////////////////////////////////////////////////////////////////////////
/// Set specified font.

void TTF::SetTextFont(Font_t fontnumber)
{
   Init();
   fgHandle->SetTextFont(fontnumber);
}

////////////////////////////////////////////////////////////////////////////////

void TTF::SetTextSize(Float_t textsize)
{
   Init();
   fgHandle->SetTextSize(textsize);
}

////////////////////////////////////////////////////////////////////////////////
/// Put the characters in "string" in the "glyphs" array.

void TTF::PrepareString(const char *string)
{
   Init();
   fgHandle->PrepareString(string);
}

////////////////////////////////////////////////////////////////////////////////
/// Put the characters in "string" in the "glyphs" array.

void TTF::PrepareString(const wchar_t *string)
{
   Init();
   fgHandle->PrepareString(string);
}

////////////////////////////////////////////////////////////////////////////////
/// Compute the glyphs positions, fgAscent and fgWidth (needed for alignment).

void TTF::LayoutGlyphs()
{
   if (fgHandle)
      fgHandle->LayoutGlyphs();
}

////////////////////////////////////////////////////////////////////////////////
/// Compute the trailing blanks width. It is use to compute the text width in GetTextExtent
/// `n` is the number of trailing blanks in a string.

void TTF::ComputeTrailingBlanksWidth(Int_t n)
{
   if (fgHandle)
      fgHandle->ComputeTrailingBlanksWidth(n);
}

////////////////////////////////////////////////////////////////////////////////
/// Remove temporary data created by LayoutGlyphs

void TTF::CleanupGlyphs()
{
   if (fgHandle)
      fgHandle->CleanupGlyphs();
}

////////////////////////////////////////////////////////////////////////////////
/// Get width (w) and height (h) when text is horizontal.

void TTF::GetTextExtent(UInt_t &w, UInt_t &h, const char *text)
{
   Init();
   fgHandle->GetTextExtent(w, h, text);
}

////////////////////////////////////////////////////////////////////////////////
/// Get advance (a) when text is horizontal.

void TTF::GetTextAdvance(UInt_t &a, const char *text)
{
   Init();
   fgHandle->GetTextAdvance(a, text);
}

////////////////////////////////////////////////////////////////////////////////
/// Get width (w) and height (h) when text is horizontal.

void TTF::GetTextExtent(UInt_t &w, UInt_t &h, const wchar_t *text)
{
   Init();
   fgHandle->GetTextExtent(w, h, text);
}

////////////////////////////////////////////////////////////////////////////////

void TTF::Version(Int_t &major, Int_t &minor, Int_t &patch)
{
   Init();
   fgHandle->Version(major, minor, patch);
}
