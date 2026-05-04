// @(#)root/graf:$Id$
// Author: Olivier Couet     01/10/2002
// Author: Fons Rademakers   21/11/1998
// Author: Sergey Linev      29/14/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTF
#define ROOT_TTF


#include "Rtypes.h"
#include <memory>

/// @cond DOXYGEN_IGNORE
// Forward declare for the headers:
// #  include <ft2build.h>
// #  include FT_FREETYPE_H
// #  include FT_GLYPH_H
extern "C" {
   struct FT_LibraryRec_;
   struct FT_FaceRec_;
   struct FT_CharMapRec_;
   struct FT_GlyphRec_;
   struct FT_Matrix_;
   struct FT_Bitmap_;
   typedef struct FT_LibraryRec_* FT_Library;
   typedef struct FT_FaceRec_* FT_Face;
   typedef struct FT_CharMapRec_* FT_CharMap;
   typedef struct FT_GlyphRec_* FT_Glyph;
   typedef struct FT_Matrix_ FT_Matrix;
   typedef struct FT_Bitmap_ FT_Bitmap; // Forward declared for TGX11TTF.h's benefit
   typedef signed long FT_Pos;
   #ifndef FT_FREETYPE_H
   struct FT_Vector_ { FT_Pos x, y; };
   struct FT_BBox_ { FT_Pos xMin, yMin, xMax, yMax; };
   #endif
   typedef struct FT_Vector_ FT_Vector;
   typedef struct FT_BBox_ FT_BBox;
}
/// @endcond


class TTFhandle;
struct TTFontHandle;

class TTF {

   friend class TTFhandle;

public:

   /** \class TTGlyph
       TTF helper class containing glyphs description.
   */

   class TTGlyph {
   public:
      UInt_t     fIndex{0};     ///< glyph index in face
      FT_Vector  fPos;          ///< position of glyph origin
      FT_Glyph   fImage{nullptr}; ///< glyph image
      TTGlyph(UInt_t indx = 0) : fIndex(indx) {}
   };

   TTF() {}
   virtual ~TTF();

   // old static methods which are fully replaced by TTFhandle
   // remain here only for backward compatibility until ROOT7

   // minimal static interface to initialize library and handle fonts
   static void           Init();
   static void           Cleanup();
   static Bool_t         IsInitialized();

   static Short_t        CharToUnicode(UInt_t code);
   static void           LayoutGlyphs();
   static void           PrepareString(const char *string);
   static void           PrepareString(const wchar_t *string);
   static void           SetRotationMatrix(Float_t angle);

   static void           ComputeTrailingBlanksWidth(Int_t n);
   static Int_t          GetAscent();
   static const FT_BBox &GetBox();
   static TTGlyph       *GetGlyphs();
   static Bool_t         GetHinting();
   static Bool_t         GetKerning();
   static Int_t          GetNumGlyphs();
   static FT_Matrix     *GetRotMatrix();
   static Bool_t         GetSmoothing();
   static Int_t          GetTrailingBlanksWidth();
   static Int_t          GetWidth();
   static void           SetHinting(Bool_t state);
   static void           SetKerning(Bool_t state);
   static void           SetSmoothing(Bool_t state);
   static void           GetTextExtent(UInt_t &w, UInt_t &h, const char *text);
   static void           GetTextExtent(UInt_t &w, UInt_t &h, const wchar_t *text);
   static void           GetTextAdvance(UInt_t &a, const char *text);
   static void           SetTextFont(Font_t fontnumber);
   static Int_t          SetTextFont(const char *fontname, Int_t italic=0);
   static void           SetTextSize(Float_t textsize);

   static void           Version(Int_t &major, Int_t &minor, Int_t &patch);

   // new temporary methods, can be removed at the end
   static void           CleanupGlyphs();

   ClassDef(TTF,0)  //Interface to TTF font handling
};

class TTFhandle {
   friend class TTF;

   private:
      TTFontHandle  *fFont = nullptr;            ///< selected font
      Int_t          fAscent = 0;                ///< string ascent, used to compute Y alignment
      FT_BBox        fCBox;                      ///< string control box
      std::vector<TTF::TTGlyph> fGlyphs;         ///< glyphs
      Bool_t         fKerning = kTRUE;           ///< use kerning (true by default)
      std::unique_ptr<FT_Matrix> fRotMatrix;     ///< rotation matrix
      Int_t          fTBlankW = 0;               ///< trailing blanks width
      Int_t          fWidth = 0;                 ///< string width, used to compute X alignment

      static  Bool_t fgHinting;                   ///< use hinting (false by default)
      static  Bool_t fgSmoothing;                 ///< use anti-aliasing (true when >8 planes, false otherwise)

      UInt_t         CharToUnicode(UInt_t code);
      void           ComputeTrailingBlanksWidth(Int_t n);

      static FT_Library InitClose(Int_t direction = 0);

      Int_t          SelectFontHandle(Int_t arg, const char *name = nullptr);

   public:
      TTFhandle();
      virtual ~TTFhandle();

      TTF::TTGlyph  *GetGlyphs() { return fGlyphs.data(); }
      UInt_t         GetNumGlyphs() const { return fGlyphs.size(); }
      FT_Face        GetFontFace() const;
      Int_t          GetAscent() const { return fAscent; }
      Bool_t         GetKerning() const { return fKerning; }
      FT_Matrix     *GetRotMatrix() const { return fRotMatrix.get(); }
      Int_t          GetTrailingBlanksWidth() const { return fTBlankW; }
      Int_t          GetWidth() const { return fWidth; }
      const FT_BBox &GetBox() const { return fCBox; }

      void           SetKerning(Bool_t state) { fKerning = state; }
      void           SetTextFont(Font_t fontnumber);
      Int_t          SetTextFont(const char *fontname, Int_t italic = 0);
      Bool_t         SetTextSize(Float_t textsize);

      void           LayoutGlyphs();
      void           PrepareString(const char *string);
      void           PrepareString(const wchar_t *string);
      void           SetRotationMatrix(Float_t angle);
      void           CleanupGlyphs();

      void           GetTextExtent(UInt_t &w, UInt_t &h, const char *text);
      void           GetTextExtent(UInt_t &w, UInt_t &h, const wchar_t *text);
      void           GetTextAdvance(UInt_t &a, const char *text);

      void           Version(Int_t &major, Int_t &minor, Int_t &patch);


      static Bool_t  Init();
      static Bool_t  GetHinting();
      static Bool_t  GetSmoothing();
      static void    SetHinting(Bool_t state);
      static void    SetSmoothing(Bool_t state);

   ClassDef(TTFhandle, 0)  // Dynamic interface to TTF

};

#endif
