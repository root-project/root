// @(#)root/graf:$Id$
// Author: Sergey Linev      25/08/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTFhandle
#define ROOT_TTFhandle


#include "Rtypes.h"
#include <memory>

class TTF;

class TTFhandle {
   friend class TTF;

   private:

      struct GlyphStruct;
      struct FontStruct;

      Bool_t         fKerning = kTRUE;           ///< use kerning (true by default)
      Bool_t         fSmoothing = kTRUE;         ///< use anti-aliasing (true when >8 planes, false otherwise)
      Bool_t         fHinting = kFALSE;          ///< use hinting (false by default)

      FontStruct    *fFont = nullptr;            ///< selected font
      long           fRotationXX = 0, fRotationXY = 0; ///< rotation matrix members
      std::vector<GlyphStruct> fGlyphs;          ///< glyphs
      Int_t          fAscent = 0;                ///< string ascent, used to compute Y alignment
      long xMin = 0, yMin = 0, xMax = 0, yMax = 0; ///< boundaries
      Int_t          fTBlankW = 0;               ///< trailing blanks width
      Int_t          fWidth = 0;                 ///< string width, used to compute X alignment

      UInt_t         CharToUnicode(UInt_t code);
      void           ComputeTrailingBlanksWidth(Int_t n);

      Int_t          SelectFontHandle(Int_t arg, const char *name = nullptr);

      /// Thread-local wrapper to the FreeType library
      struct FT_Library_Wrapper;
      static thread_local FT_Library_Wrapper fFT_Library;

      void FillTTFGlypths(void *vect);

   public:
      TTFhandle();
      virtual ~TTFhandle();

      void           SetSmoothing(Bool_t state) { fSmoothing = state;  }
      void           SetHinting(Bool_t state) { fHinting = state; }
      void           SetKerning(Bool_t state) { fKerning = state; }
      void           SetTextFont(Font_t fontnumber);
      Int_t          SetTextFont(const char *fontname, Int_t italic = 0);
      Bool_t         SetTextSize(Float_t textsize);
      void           SetRotationMatrix(Float_t angle);

      Bool_t         GetSmoothing() const { return fSmoothing; }
      Bool_t         GetHinting() const { return fHinting; }
      Bool_t         GetKerning() const { return fKerning; }
      void*          GetFontFace() const;

      void           PrepareString(const char *string);
      void           PrepareString(const wchar_t *string);
      void           LayoutGlyphs();

      Int_t          GetGlyphsWidth() const;
      Int_t          GetGlyphsHeight() const;

      Int_t          GetAscent() const { return fAscent; }
      Int_t          GetTrailingBlanksWidth() const { return fTBlankW; }
      Int_t          GetWidth() const { return fWidth; }
      long           GetBoxXMin() const { return xMin; }
      long           GetBoxXMax() const { return xMax; }
      long           GetBoxYMin() const { return yMin; }
      long           GetBoxYMax() const { return yMax; }

      UInt_t         GetNumGlyphs() const;
      Bool_t         ApplyAlignRotate(Int_t &px, Int_t &py, Int_t align, Int_t pad_width, Int_t pad_height);
      Bool_t         GetGlyphData(UInt_t n, Int_t &offx, Int_t &offy, UChar_t *&buffer, UInt_t &width, UInt_t &rows, UInt_t &pitch);
      void           CleanupGlyphs();

      void           GetTextExtent(UInt_t &w, UInt_t &h, const char *text);
      void           GetTextExtent(UInt_t &w, UInt_t &h, const wchar_t *text);
      void           GetTextAdvance(UInt_t &a, const char *text);

      void           Version(Int_t &major, Int_t &minor, Int_t &patch);

      static Bool_t  Init();

   ClassDef(TTFhandle, 0)  // Dynamic interface to TTF

};

#endif
