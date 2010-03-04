// @(#)root/graf:$Id$
// Author: Olivier Couet     01/10/02
// Author: Fons Rademakers   21/11/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTF
#define ROOT_TTF


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTF                                                                  //
//                                                                      //
// Interface to the Freetype 2 library.                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#ifndef __CINT__
#  include <ft2build.h>
#  include FT_FREETYPE_H
#  include FT_GLYPH_H
#else
   typedef void* FT_Library;
   typedef void* FT_Face;
   typedef void* FT_CharMap;
   typedef void* FT_Glyph;
   typedef signed long FT_Pos;
   struct FT_Vector { FT_Pos x, y; };
   struct FT_BBox { FT_Pos xMin, yMin, xMax, yMax; };
   struct FT_Matrix;
   struct FT_Bitmap;
#endif


// Class (actually structure) containing glyphs description
class TTGlyph {
public:
   UInt_t     fIndex; // glyph index in face
   FT_Vector  fPos;   // position of glyph origin
   FT_Glyph   fImage; // glyph image
};


class TGX11TTF;
class TGWin32;


class TTF {

friend class TGX11TTF;
friend class TGWin32;

private:
   enum { kTTMaxFonts = 32, kMaxGlyphs = 1024 };

   static Int_t       fgAscent;                // string ascent, used to compute Y alignment
   static FT_BBox     fgCBox;                  // string control box
   static FT_CharMap  fgCharMap[kTTMaxFonts];  // font character map
   static Int_t       fgCurFontIdx;            // current font index
   static Int_t       fgSymbItaFontIdx;        // Symbol italic font index
   static Int_t       fgFontCount;             // number of fonts loaded
   static char       *fgFontName[kTTMaxFonts]; // font name
   static FT_Face     fgFace[kTTMaxFonts];     // font face
   static TTGlyph     fgGlyphs[kMaxGlyphs];    // glyphs
   static Bool_t      fgHinting;               // use hinting (true by default)
   static Bool_t      fgInit;                  // true if the Init has been called
   static Bool_t      fgKerning;               // use kerning (true by default)
   static FT_Library  fgLibrary;               // FreeType font library
   static Int_t       fgNumGlyphs;             // number of glyphs in the string
   static FT_Matrix  *fgRotMatrix;             // rotation matrix
   static Bool_t      fgSmoothing;             // use anti-aliasing (true when >8 planes, false otherwise)
   static Int_t       fgTBlankW;               // trailing blanks width
   static Int_t       fgWidth;                 // string width, used to compute X alignment

public:
   static Short_t CharToUnicode(UInt_t code);
   static void    LayoutGlyphs();
   static void    PrepareString(const char *string);
   static void    SetRotationMatrix(Float_t angle);

public:
   TTF() { }
   virtual ~TTF();

   static void           Init();
   static void           Cleanup();
   static Int_t          GetAscent();
   static const FT_BBox &GetBox();
   static TTGlyph       *GetGlyphs();
   static Bool_t         GetHinting();
   static Bool_t         GetKerning();
   static Int_t          GetNumGlyphs();
   static FT_Matrix     *GetRotMatrix();
   static Bool_t         GetSmoothing();
   static Int_t          GetWidth();
   static void           SetHinting(Bool_t state);
   static void           SetKerning(Bool_t state);
   static void           SetSmoothing(Bool_t state);
   static void           GetTextExtent(UInt_t &w, UInt_t &h, char *text);
   static void           GetTextAdvance(UInt_t &a, char *text);
   static void           SetTextFont(Font_t fontnumber);
   static Int_t          SetTextFont(const char *fontname, Int_t italic=0);
   static void           SetTextSize(Float_t textsize);
   static Bool_t         IsInitialized();
   static void           Version(Int_t &major, Int_t &minor, Int_t &patch);

   ClassDef(TTF,0)  //Interface to TTF font handling
};

#endif
