// @(#)root/graf:$Id$
// Author: Sergey  Linev     29/04/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


/** \class TTFhandle
\ingroup BasicGraphics

Dynamic handle to work with freetype 2 library.
in ROOT7 TTFhandle will be renamed into TTF class
*/


#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_GLYPH_H

#include "TROOT.h"
#include "TTF.h"
#include "TSystem.h"
#include "TEnv.h"
#include "TMath.h"
#include "TError.h"

// to scale fonts to the same size as the old TT version
const Float_t kScale = 0.93376068;

Bool_t TTFhandle::fgHinting = kFALSE;

struct TTFhandle::GlyphStruct {
public:
   UInt_t     fIndex{0};     ///< glyph index in face
   FT_Vector  fPos;          ///< position of glyph origin
   FT_Glyph   fImage{nullptr}; ///< glyph image
   GlyphStruct(UInt_t indx = 0) : fIndex(indx) {}
   ~GlyphStruct() { FT_Done_Glyph(fImage); }
};


struct TTFontHandle {
   std::string name;
   FT_Face face = nullptr;
   FT_CharMap charmap = nullptr;
   bool is_symbol() const
   {
      return (name == "wingding.ttf") || (name.find("symbol.ttf") == 0);
   }
};

////////////////////////////////////////////////////////////////////////////////
/// Thread-local wrapper to the freetype library.
/// The library gets initialised on demand when Get() is called.
/// It auto-destructs when the thread exits.
struct TTFhandle::FT_Library_Wrapper {
   FT_Library _library = nullptr;
   FT_Library_Wrapper() = default;
   FT_Library_Wrapper(FT_Library_Wrapper const &) = delete;
   FT_Library_Wrapper(FT_Library_Wrapper &&) = delete;
   FT_Library_Wrapper &operator=(FT_Library_Wrapper const &) = delete;
   FT_Library_Wrapper &operator=(FT_Library_Wrapper &&) = delete;

   FT_Library Get()
   {
      if (!_library && FT_Init_FreeType(&_library) != 0) {
         Error("TTF.cxx", "error initializing FreeType");
         _library = nullptr;
      }
      return _library;
   }

   ~FT_Library_Wrapper()
   {
      if (_library)
         FT_Done_FreeType(_library);
   }
};

thread_local TTFhandle::FT_Library_Wrapper TTFhandle::fFT_Library;

////////////////////////////////////////////////////////////////////////////////

TTFhandle::TTFhandle()
{
   // Ensure that there's a freetype library in our thread
   fFT_Library.Get();
}

////////////////////////////////////////////////////////////////////////////////

TTFhandle::~TTFhandle()
{
   CleanupGlyphs();
}

////////////////////////////////////////////////////////////////////////////////
/// Map char to unicode. Returns 0 in case no mapping exists.

UInt_t TTFhandle::CharToUnicode(UInt_t code)
{
   FT_Face face = fFont ? fFont->face : nullptr;
   if (!face)
      return 0;

   if (!fFont->charmap) {
      Int_t n = face->num_charmaps;
      for (Int_t i = 0; i < n; i++) {
         FT_CharMap charmap  = face->charmaps[i];
         auto platform = charmap->platform_id;
         auto encoding = charmap->encoding_id;
         if ((platform == 3 && encoding == 1) ||
             (platform == 0 && encoding == 0) ||
             (platform == 1 && encoding == 0 && fFont->is_symbol()))
         {
            fFont->charmap = charmap;
            if (FT_Set_Charmap(face, charmap))
               Error("TTF::CharToUnicode", "error in FT_Set_CharMap");
            break;
         }
      }
   }
   return FT_Get_Char_Index(face, (FT_ULong)code);
}

////////////////////////////////////////////////////////////////////////////////
/// Compute the trailing blanks width. It is use to compute the text width in GetTextExtent
/// `n` is the number of trailing blanks in a string.

void TTFhandle::ComputeTrailingBlanksWidth(Int_t n)
{
   fTBlankW = 0;
   if (n && fFont) {
      FT_Face face = fFont->face;
      char space = ' ';
      FT_UInt load_flags = FT_LOAD_DEFAULT;
      if (!fgHinting) load_flags |= FT_LOAD_NO_HINTING;
      FT_Load_Char(face, space, load_flags);

      FT_GlyphSlot slot      = face->glyph;
      FT_Pos advance_x       = slot->advance.x;
      Int_t advance_x_pixels = advance_x >> 6;

      fTBlankW = advance_x_pixels * n;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Get width (w) and height (h) when text is horizontal.

void TTFhandle::GetTextExtent(UInt_t &w, UInt_t &h, const char *text)
{
   SetRotationMatrix(0);
   PrepareString(text);
   LayoutGlyphs();
   Int_t Xoff = 0; if (xMin < 0) Xoff = -xMin;
   Int_t Yoff = 0; if (yMin < 0) Yoff = -yMin;
   w = xMax + Xoff + GetTrailingBlanksWidth();
   h = yMax + Yoff;
   CleanupGlyphs();
}

////////////////////////////////////////////////////////////////////////////////
/// Get advance (a) when text is horizontal.

void TTFhandle::GetTextAdvance(UInt_t &a, const char *text)
{
   SetRotationMatrix(0);
   PrepareString(text);
   LayoutGlyphs();
   a = GetWidth() >> 6;
   CleanupGlyphs();
}

////////////////////////////////////////////////////////////////////////////////
/// Get width (w) and height (h) when text is horizontal.

void TTFhandle::GetTextExtent(UInt_t &w, UInt_t &h, const wchar_t *text)
{
   SetRotationMatrix(0);
   PrepareString(text);
   LayoutGlyphs();
   Int_t Xoff = 0; if (xMin < 0) Xoff = -xMin;
   Int_t Yoff = 0; if (yMin < 0) Yoff = -yMin;
   w = xMax + Xoff + GetTrailingBlanksWidth();
   h = yMax + Yoff;
   CleanupGlyphs();
}

////////////////////////////////////////////////////////////////////////////////
/// Compute the glyphs positions, fgAscent and fgWidth (needed for alignment).
/// Perform the Glyphs transformation.
/// Compute the string control box.
/// If required take the "kerning" into account.
/// SetRotation and PrepareString should have been called before.

void TTFhandle::LayoutGlyphs()
{
   FT_Vector origin;
   FT_UInt   load_flags;
   FT_UInt   prev_index = 0;

   fAscent = 0;
   fWidth  = 0;

   load_flags = FT_LOAD_DEFAULT;
   if (!fgHinting)
      load_flags |= FT_LOAD_NO_HINTING;

   xMin = yMin =  32000;
   xMax = yMax = -32000;

   FT_Face face = fFont ? fFont->face : nullptr;
   if (!face)
      return;

   for (auto &glyph : fGlyphs) {

      // compute glyph origin
      if (fKerning) {
         if (prev_index) {
            FT_Vector  kern;
            FT_Get_Kerning(face, prev_index, glyph.fIndex,
                           fgHinting ? ft_kerning_default : ft_kerning_unfitted,
                           &kern);
            fWidth += kern.x;
         }
         prev_index = glyph.fIndex;
      }

      origin.x = fWidth;
      origin.y = 0;

      // clear existing image if there is one
      if (glyph.fImage) {
         FT_Done_Glyph(glyph.fImage);
         glyph.fImage = nullptr;
      }

      // load the glyph image (in its native format)
      if (FT_Load_Glyph(face, glyph.fIndex, load_flags))
         continue;

      // extract the glyph image
      if (FT_Get_Glyph(face->glyph, &glyph.fImage))
         continue;

      glyph.fPos = origin;
      fWidth    += face->glyph->advance.x;
      fAscent    = TMath::Max((Int_t)(face->glyph->metrics.horiBearingY), fAscent);

      // transform the glyphs
      FT_Matrix m, *matrix_arg = nullptr;

      if (fRotationXX || fRotationXY) {
         m.xx = m.yy = fRotationXX;
         m.xy = fRotationXY;
         m.yx = -fRotationXY;
         matrix_arg = &m;
      }

      FT_Vector_Transform(&glyph.fPos, matrix_arg);
      if (FT_Glyph_Transform(glyph.fImage, matrix_arg, &glyph.fPos))
         continue;

      // compute the string control box
      FT_BBox  bbox;
      FT_Glyph_Get_CBox(glyph.fImage, ft_glyph_bbox_pixels, &bbox);
      if (bbox.xMin < xMin) xMin = bbox.xMin;
      if (bbox.yMin < yMin) yMin = bbox.yMin;
      if (bbox.xMax > xMax) xMax = bbox.xMax;
      if (bbox.yMax > yMax) yMax = bbox.yMax;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// return number of glyphs

UInt_t TTFhandle::GetNumGlyphs() const
{
   return fGlyphs.size();
}


////////////////////////////////////////////////////////////////////////////////

void TTFhandle::FillTTFGlypths(void *data)
{
   auto &vect = *((std::vector<TTF::TTGlyph> *) data);
   vect.resize(fGlyphs.size());
   for (std::size_t i = 0; i < fGlyphs.size(); ++i) {
      auto &tgt = vect[i];
      auto &src = fGlyphs[i];

      tgt.fIndex = src.fIndex;
      tgt.fPos = src.fPos;
      tgt.fImage = src.fImage;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Apply align and configured rotation matrix to text position
/// px and py will be shifted to the place where glyph drawing can be started
/// Method returns false when glyphs not need to be drawn
/// while position is outside of specified pad dimentsions

Bool_t TTFhandle::ApplyAlignRotate(Int_t &px, Int_t &py, Int_t align, Int_t pad_width, Int_t pad_height)
{
   Int_t txalh = align / 10;
   Int_t txalv = align % 10;

   FT_Vector alignVector;

   switch (txalh) {
      case 2: alignVector.x = GetWidth() / 2; break; //center
      case 3: alignVector.x = GetWidth(); break; //right
      default: alignVector.x = 0; break; // left
   }

   switch (txalv) {
      case 2: alignVector.y = GetAscent() / 2; break; // middle
      case 3: alignVector.y = GetAscent(); break; //top
      default: alignVector.y = 0; break; //bottom
   }

   FT_Matrix m, *matrix_arg = nullptr;

   if (fRotationXX || fRotationXY) {
      m.xx = m.yy = fRotationXX;
      m.xy = fRotationXY;
      m.yx = -fRotationXY;
      matrix_arg = &m;
   }

   FT_Vector_Transform(&alignVector, matrix_arg);

   Int_t Xoff = TMath::Max(0, (Int_t) -xMin);
   Int_t Yoff = TMath::Max(0, (Int_t) -yMin);
   Int_t w    = xMax + Xoff;
   Int_t h    = yMax + Yoff;

   // If w or h is 0, very likely the string is only blank characters
   if (w <= 0 || h <= 0)
      return kFALSE;

   Int_t x1 = px - Xoff - (alignVector.x >> 6);
   Int_t y1 = py + Yoff + (alignVector.y >> 6) - h;

   // If string falls outside window, there is probably no need to draw it.
   if (x1 + w <= 0 || x1 >= pad_width || y1 + h <= 0 || y1 >= pad_height)
      return kFALSE;

   // do not draw text, which size is significantly larger than available pad
   if ((w > 10 * pad_width) || (h > 10 * pad_height))
      return kFALSE;

   px = x1;
   py = y1;
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns width of all glyphs

Int_t TTFhandle::GetGlyphsWidth() const
{
   return xMax + TMath::Max(0, (Int_t) -xMin);
}

////////////////////////////////////////////////////////////////////////////////
/// Returns height of all glyphs

Int_t TTFhandle::GetGlyphsHeight() const
{
   return yMax + TMath::Max(0, (Int_t) -yMin);
}

////////////////////////////////////////////////////////////////////////////////
/// Returns data for glyph bitmap
/// Instead direct access to FT_BitmapGlyph one can obtain all relevant fields
/// Thus one do not requires work with TrueType classes directly
/// Return kFALSE when glyph not exists or if it width is zero

Bool_t TTFhandle::GetGlyphData(UInt_t n, Int_t &offx, Int_t &offy, UChar_t *&buffer, UInt_t &width, UInt_t &rows, UInt_t &pitch)
{
   if (n >= fGlyphs.size())
      return kFALSE;

   if (FT_Glyph_To_Bitmap(&fGlyphs[n].fImage, GetSmoothing() ? ft_render_mode_normal : ft_render_mode_mono, nullptr, 1))
      return kFALSE;

   auto glyph = fGlyphs[n].fImage;
   if (!glyph || (glyph->format != FT_GLYPH_FORMAT_BITMAP))
      return kFALSE;

   // 2. Safe to typecast to FT_BitmapGlyph
   FT_BitmapGlyph bitmap_glyph = (FT_BitmapGlyph)glyph;

   auto &bmp = bitmap_glyph->bitmap;
   if (!bmp.width)
      return kFALSE;

   offx = TMath::Max(0, (Int_t) -xMin) + bitmap_glyph->left;
   offy = yMax - bitmap_glyph->top;

   buffer = bmp.buffer;
   width = bmp.width;
   rows = bmp.rows;
   pitch = bmp.pitch;
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Remove temporary data created by LayoutGlyphs

void TTFhandle::CleanupGlyphs()
{
   fGlyphs.clear();
}

////////////////////////////////////////////////////////////////////////////////
/// Put the characters in "string" in the "glyphs" array.

void TTFhandle::PrepareString(const char *string)
{
   CleanupGlyphs();

   const unsigned char *p = (const unsigned char*) string;

   Int_t NbTBlank = 0; // number of trailing blanks

   while (*p) {
      UInt_t index = CharToUnicode((FT_ULong)*p);
      if (index != 0)
         fGlyphs.emplace_back(index);
      if (*p == ' ')
         NbTBlank++;
      else
         NbTBlank = 0;
      p++;
   }

   ComputeTrailingBlanksWidth(NbTBlank);
}

////////////////////////////////////////////////////////////////////////////////
/// Put the characters in "string" in the "glyphs" array.

void TTFhandle::PrepareString(const wchar_t *string)
{
   CleanupGlyphs();

   FT_Face face = fFont ? fFont->face : nullptr;
   if (!face)
      return;

   const wchar_t *p = string;

   Int_t NbTBlank = 0; // number of trailing blanks

   while (*p) {
      UInt_t index = FT_Get_Char_Index(face, (FT_ULong) *p);
      if (index != 0)
         fGlyphs.emplace_back(index);
      if (*p == ' ')
         NbTBlank++;
      else
         NbTBlank = 0;
      p++;
   }

   ComputeTrailingBlanksWidth(NbTBlank);
}

////////////////////////////////////////////////////////////////////////////////
/// Return current font index

void *TTFhandle::GetFontFace() const
{
   return fFont ? (void *) fFont->face : nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Set the rotation matrix used to rotate the font outlines.

void TTFhandle::SetRotationMatrix(Float_t angle)
{
   fRotationXX = fRotationXY = 0;
   if (!angle)
      return;

   Float_t rangle = angle * TMath::Pi() / 180.; // Angle in radian
#if defined(FREETYPE_PATCH) && \
    (FREETYPE_MAJOR == 2) && (FREETYPE_MINOR == 1) && (FREETYPE_PATCH == 2)
   Float_t sin    = TMath::Sin(rangle);
   Float_t cos    = TMath::Cos(rangle);
#else
   Float_t sin    = TMath::Sin(-rangle);
   Float_t cos    = TMath::Cos(-rangle);
#endif

   fRotationXX = (FT_Fixed) (cos * (1<<16));
   fRotationXY = (FT_Fixed) (sin * (1<<16));

//   fRotMatrix->xx = (FT_Fixed) (cos * (1<<16));
//   fRotMatrix->xy = (FT_Fixed) (sin * (1<<16));
//   fRotMatrix->yx = -fRotMatrix->xy;
//   fRotMatrix->yy =  fRotMatrix->xx;
}

////////////////////////////////////////////////////////////////////////////////
/// Return thread_local instance of TTFontHandle for speified font

Int_t TTFhandle::SelectFontHandle(Int_t arg, const char *name)
{
   thread_local std::map<std::string, TTFontHandle> _fonts;

   fFont = nullptr;

   if (arg == 111) {
      // select any existing font, fallback solution for some errors in SetTextFont
      if (!_fonts.empty())
         fFont =  &(_fonts.begin()->second);
      Warning("TTFhandle::SetTextFont", "%s, using %s", name, fFont ? fFont->name.c_str() : "<nothing>");
      return fFont ? 0 : 1;
   }

   if (arg >= 0) {
      auto iter = _fonts.find(name);
      if (iter != _fonts.end()) {
         fFont = &iter->second;
         return 0;
      }
      if (arg == 0)
         return 1;
      _fonts[name] = { name, nullptr, nullptr };
      fFont = &_fonts[name];
      return 0;
   }

   for (auto &font : _fonts) {
      if (font.second.face) {
         FT_Done_Face(font.second.face);
         font.second.face = nullptr;
      }
   }
   _fonts.clear();
   return 0;
}


////////////////////////////////////////////////////////////////////////////////
/// Set text font to specified name.
///  - font       : font name
///  - italic     : the fonts should be slanted. Used for symbol font.
///
/// Set text font to specified name. This function returns 0 if
/// the specified font is found, 1 if not.

Int_t TTFhandle::SetTextFont(const char *fontname, Int_t italic)
{
   fFont = nullptr;

   if (!fontname || !*fontname)
      return SelectFontHandle(111, "no font name specified");

   const char *basename = gSystem->BaseName(fontname);

   if (SelectFontHandle(1, TString::Format("%s%s", basename, italic ? ".italic" : ""))) {
      Fatal("SetTextFont", "Fail to create font handle for font %s", basename);
      return 1;
   }

   // font face exists and initialized
   if (fFont->face)
      return 0;

   auto lib = fFT_Library.Get();
   if (!lib) {
      Error("SetTextFont", "no free type library initialized");
      return 1;
   }

   // try to load font (font must be in Root.TTFontPath resource)
   const char *ttpath = gEnv->GetValue("Root.TTFontPath", TROOT::GetTTFFontDir());

   TString fname = fontname;
   const char *ttfont = gSystem->FindFile(ttpath, fname, kReadPermission);

   if (!ttfont)
      return SelectFontHandle(111, TString::Format("font file %s not found in path %s", fontname, ttpath));

   if (FT_New_Face(lib, ttfont, 0, &fFont->face))
      return SelectFontHandle(111, TString::Format("error loading font %s", ttfont));

   if (italic) {
      FT_Matrix slantMat;
      slantMat.xx = (1 << 16);
      slantMat.xy = ((1 << 16) >> 2);
      slantMat.yx = 0;
      slantMat.yy = (1 << 16);
      FT_Set_Transform(fFont->face, &slantMat, nullptr);
   }

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Set specified font.
/// List of the currently supported fonts (screen and PostScript)
///
/// | Font number |     TTF Names             |     PostScript/PDF Names      |
/// |-------------|---------------------------|-------------------------------|
/// |      1      |   Free Serif Italic       |    Times-Italic               |
/// |      2      |   Free Serif Bold         |    Times-Bold                 |
/// |      3      |   Free Serif Bold Italic  |    Times-BoldItalic           |
/// |      4      |   Tex Gyre Regular        |    Helvetica                  |
/// |      5      |   Tex Gyre Italic         |    Helvetica-Oblique          |
/// |      6      |   Tex Gyre Bold           |    Helvetica-Bold             |
/// |      7      |   Tex Gyre Bold Italic    |    Helvetica-BoldOblique      |
/// |      8      |   Free Mono               |    Courier                    |
/// |      9      |   Free Mono Oblique       |    Courier-Oblique            |
/// |     10      |   Free Mono Bold          |    Courier-Bold               |
/// |     11      |   Free Mono Bold Oblique  |    Courier-BoldOblique        |
/// |     12      |   Symbol                  |    Symbol                     |
/// |     13      |   Free Serif              |    Times-Roman                |
/// |     14      |   Wingdings               |    ZapfDingbats               |

void TTFhandle::SetTextFont(Font_t fontnumber)
{
   // Added by cholm for use of DFSG - fonts - based on Kevins fix.
   // Table of Microsoft and (for non-MSFT operating systems) backup
   // FreeFont TTF fonts.
   static const char *fonttable[][2] = {
     { "Root.TTFont.0", "FreeSansBold.otf" },
     { "Root.TTFont.1", "FreeSerifItalic.otf" },
     { "Root.TTFont.2", "FreeSerifBold.otf" },
     { "Root.TTFont.3", "FreeSerifBoldItalic.otf" },
     { "Root.TTFont.4", "texgyreheros-regular.otf" },
     { "Root.TTFont.5", "texgyreheros-italic.otf" },
     { "Root.TTFont.6", "texgyreheros-bold.otf" },
     { "Root.TTFont.7", "texgyreheros-bolditalic.otf" },
     { "Root.TTFont.8", "FreeMono.otf" },
     { "Root.TTFont.9", "FreeMonoOblique.otf" },
     { "Root.TTFont.10", "FreeMonoBold.otf" },
     { "Root.TTFont.11", "FreeMonoBoldOblique.otf" },
     { "Root.TTFont.12", "symbol.ttf" },
     { "Root.TTFont.13", "FreeSerif.otf" },
     { "Root.TTFont.14", "wingding.ttf" },
     { "Root.TTFont.15", "symbol.ttf" },
     { "Root.TTFont.STIXGen", "STIXGeneral.otf" },
     { "Root.TTFont.STIXGenIt", "STIXGeneralItalic.otf" },
     { "Root.TTFont.STIXGenBd", "STIXGeneralBol.otf" },
     { "Root.TTFont.STIXGenBdIt", "STIXGeneralBolIta.otf" },
     { "Root.TTFont.STIXSiz1Sym", "STIXSiz1Sym.otf" },
     { "Root.TTFont.STIXSiz1SymBd", "STIXSiz1SymBol.otf" },
     { "Root.TTFont.STIXSiz2Sym", "STIXSiz2Sym.otf" },
     { "Root.TTFont.STIXSiz2SymBd", "STIXSiz2SymBol.otf" },
     { "Root.TTFont.STIXSiz3Sym", "STIXSiz3Sym.otf" },
     { "Root.TTFont.STIXSiz3SymBd", "STIXSiz3SymBol.otf" },
     { "Root.TTFont.STIXSiz4Sym", "STIXSiz4Sym.otf" },
     { "Root.TTFont.STIXSiz4SymBd", "STIXSiz4SymBol.otf" },
     { "Root.TTFont.STIXSiz5Sym", "STIXSiz5Sym.otf" },
     { "Root.TTFont.ME", "DroidSansFallback.ttf" },
     { "Root.TTFont.CJKMing", "DroidSansFallback.ttf" },
     { "Root.TTFont.CJKGothic", "DroidSansFallback.ttf" }
   };

   static int fontset = -1;
   int        thisset = fontset;

   int fontid = fontnumber / 10;
   if (fontid < 0 || fontid > 31) fontid = 0;

   if (thisset == -1) {
      // try to load font (font must be in Root.TTFontPath resource)
      // to see which fontset we have available
      const char *ttpath = gEnv->GetValue("Root.TTFontPath",
                                          TROOT::GetTTFFontDir());
      TString fname = gEnv->GetValue(fonttable[fontid][0], fonttable[fontid][1]);
      const char *ttfont = gSystem->FindFile(ttpath, fname, kReadPermission);
      thisset = ttfont ? 0 : 1;
   }
   Int_t italic = fontid == 15 ? 1 : 0;
   auto ret = SetTextFont(gEnv->GetValue(fonttable[fontid][thisset], fonttable[fontid][1]), italic);

   // Do not define font set is we're loading the symbol.ttf - it's
   // the same in both cases.
   if (ret == 0 && fontid != 12)
      fontset = thisset;
}

////////////////////////////////////////////////////////////////////////////////
/// Set current text size.

Bool_t TTFhandle::SetTextSize(Float_t textsize)
{
   if (textsize < 0)
      return kFALSE;

   if (!fFont || !fFont->face) {
      Error("TTFhandle::SetTextSize", "current font not selected");
      return kFALSE;
   }

   Int_t tsize = (Int_t)(textsize*kScale+0.5) << 6;
   FT_Error err = FT_Set_Char_Size(fFont->face, tsize, tsize, 72, 72);

   if (err)
      Error("TTFhandle::SetTextSize", "error in FT_Set_Char_Size: 0x%x (input size %f, calc. size 0x%x)", err, textsize, tsize);

   return !err;
}

////////////////////////////////////////////////////////////////////////////////

void TTFhandle::Version(Int_t &major, Int_t &minor, Int_t &patch)
{
   FT_Library_Version(fFT_Library.Get(), &major, &minor, &patch);
}


////////////////////////////////////////////////////////////////////////////////

Bool_t TTFhandle::Init()
{
   return fFT_Library.Get() != nullptr;
}


////////////////////////////////////////////////////////////////////////////////

Bool_t TTFhandle::GetHinting()
{
   return fgHinting;
}


////////////////////////////////////////////////////////////////////////////////

void TTFhandle::SetHinting(Bool_t state)
{
   fgHinting = state;
}


