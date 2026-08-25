// @(#)root/graf:$Id$
// Author: Sergey Linev      29/04/2026

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
class TTFontHandle;

class TTFhandle {
   friend class TTF;

   private:

      struct GlyphStruct;

      TTFontHandle  *fFont = nullptr;            ///< selected font
      Int_t          fAscent = 0;                ///< string ascent, used to compute Y alignment
      // FT_BBox        fCBox;                      ///< string control box
      long xMin = 0, yMin = 0, xMax = 0, yMax = 0; ///< boundaries
      std::vector<GlyphStruct> fGlyphs;          ///< glyphs
      Bool_t         fKerning = kTRUE;           ///< use kerning (true by default)
      Bool_t         fSmoothing = kTRUE;         ///< use anti-aliasing (true when >8 planes, false otherwise)
      long           fRotationXX = 0, fRotationXY = 0;
      Int_t          fTBlankW = 0;               ///< trailing blanks width
      Int_t          fWidth = 0;                 ///< string width, used to compute X alignment

      static  Bool_t fgHinting;                   ///< use hinting (false by default)


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

      UInt_t         GetNumGlyphs() const;
      void*          GetFontFace() const;
      Int_t          GetAscent() const { return fAscent; }

      void           SetSmoothing(Bool_t state) { fSmoothing = state;  }
      Bool_t         GetSmoothing() const { return fSmoothing; }

      Bool_t         GetKerning() const { return fKerning; }
      Int_t          GetTrailingBlanksWidth() const { return fTBlankW; }
      Int_t          GetWidth() const { return fWidth; }

      long           GetBoxXMin() const { return xMin; }
      long           GetBoxXMax() const { return xMax; }
      long           GetBoxYMin() const { return yMin; }
      long           GetBoxYMax() const { return yMax; }

      void           SetKerning(Bool_t state) { fKerning = state; }
      void           SetTextFont(Font_t fontnumber);
      Int_t          SetTextFont(const char *fontname, Int_t italic = 0);
      Bool_t         SetTextSize(Float_t textsize);

      void           LayoutGlyphs();
      void           PrepareString(const char *string);
      void           PrepareString(const wchar_t *string);
      void           SetRotationMatrix(Float_t angle);
      Int_t          GetGlyphsWidth() const;
      Int_t          GetGlyphsHeight() const;
      Bool_t         ApplyAlignRotate(Int_t &px, Int_t &py, Int_t align, Int_t pad_width, Int_t pad_height);
      Bool_t         GetGlyphData(UInt_t n, Int_t &offx, Int_t &offy, UChar_t *&buffer, UInt_t &width, UInt_t &rows, UInt_t &pitch);
      void           CleanupGlyphs();

      void           GetTextExtent(UInt_t &w, UInt_t &h, const char *text);
      void           GetTextExtent(UInt_t &w, UInt_t &h, const wchar_t *text);
      void           GetTextAdvance(UInt_t &a, const char *text);

      void           Version(Int_t &major, Int_t &minor, Int_t &patch);

      static Bool_t  Init();
      static Bool_t  GetHinting();
      static void    SetHinting(Bool_t state);

   ClassDef(TTFhandle, 0)  // Dynamic interface to TTF

};

#endif
