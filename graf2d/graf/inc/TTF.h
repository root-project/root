// @(#)root/graf:$Id$
// Author: Olivier Couet     01/10/2002
// Author: Fons Rademakers   21/11/1998
// Author: Sergey Linev      29/04/2026

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
   struct FT_BitmapGlyphRec_;
   typedef struct FT_LibraryRec_* FT_Library;
   typedef struct FT_FaceRec_* FT_Face;
   typedef struct FT_CharMapRec_* FT_CharMap;
   typedef struct FT_GlyphRec_* FT_Glyph;
   typedef struct FT_Matrix_ FT_Matrix;
   typedef struct FT_Bitmap_ FT_Bitmap; // Forward declared for TGX11TTF.h's benefit
   typedef struct FT_BitmapGlyphRec_* FT_BitmapGlyph;
   typedef signed long FT_Pos;
   #ifndef FT_FREETYPE_H
   struct FT_Vector_ { FT_Pos x, y; };
   struct FT_BBox_ { FT_Pos xMin, yMin, xMax, yMax; };
   #endif
   typedef struct FT_Vector_ FT_Vector;
   typedef struct FT_BBox_ FT_BBox;
}
/// @endcond


#include "TTFhandle.h"



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
      ~TTGlyph() {}
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

#endif
