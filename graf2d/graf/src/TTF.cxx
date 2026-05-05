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
Bool_t TTFhandle::fgSmoothing = kTRUE;


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

TTFhandle::TTFhandle()
{
   InitClose(1);
}


////////////////////////////////////////////////////////////////////////////////

TTFhandle::~TTFhandle()
{
   CleanupGlyphs();
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize or close FreeType library
/// If argument is 0 - just return current handle
/// Library initialized per thread

FT_Library TTFhandle::InitClose(Int_t direction)
{
   thread_local FT_Library _library = nullptr;
   if ((direction > 0) || !_library) {
      if (FT_Init_FreeType(&_library)) {
         Error("TTFhandle::InitClose", "error initializing FreeType");
         _library = nullptr;
      }
   } else if ((direction < 0) && _library) {
      // SelectFontHandle(-1); // delete all font handles
      FT_Done_FreeType(_library);
      _library = nullptr;
   }

   return _library;
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
   Int_t Xoff = 0; if (fCBox.xMin < 0) Xoff = -fCBox.xMin;
   Int_t Yoff = 0; if (fCBox.yMin < 0) Yoff = -fCBox.yMin;
   w = fCBox.xMax + Xoff + GetTrailingBlanksWidth();
   h = fCBox.yMax + Yoff;
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
   Int_t Xoff = 0; if (fCBox.xMin < 0) Xoff = -fCBox.xMin;
   Int_t Yoff = 0; if (fCBox.yMin < 0) Yoff = -fCBox.yMin;
   w = fCBox.xMax + Xoff + GetTrailingBlanksWidth();
   h = fCBox.yMax + Yoff;
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

   fCBox.xMin = fCBox.yMin =  32000;
   fCBox.xMax = fCBox.yMax = -32000;

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
      FT_Vector_Transform(&glyph.fPos, fRotMatrix.get());
      if (FT_Glyph_Transform(glyph.fImage, fRotMatrix.get(), &glyph.fPos))
         continue;

      // compute the string control box
      FT_BBox  bbox;
      FT_Glyph_Get_CBox(glyph.fImage, ft_glyph_bbox_pixels, &bbox);
      if (bbox.xMin < fCBox.xMin) fCBox.xMin = bbox.xMin;
      if (bbox.yMin < fCBox.yMin) fCBox.yMin = bbox.yMin;
      if (bbox.xMax > fCBox.xMax) fCBox.xMax = bbox.xMax;
      if (bbox.yMax > fCBox.yMax) fCBox.yMax = bbox.yMax;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Return bitmap for specified glyph

FT_BitmapGlyph TTFhandle::GetGlyphBitmap(UInt_t n, Bool_t smooth)
{
   if (n >= fGlyphs.size())
      return nullptr;

   if (FT_Glyph_To_Bitmap(&fGlyphs[n].fImage, smooth || GetSmoothing() ? ft_render_mode_normal : ft_render_mode_mono, nullptr, 1))
      return nullptr;

   return (FT_BitmapGlyph) fGlyphs[n].fImage;
}

////////////////////////////////////////////////////////////////////////////////
/// Remove temporary data created by LayoutGlyphs

void TTFhandle::CleanupGlyphs()
{
   bool is_lib = InitClose() != nullptr;

   for(auto &glyph : fGlyphs) {
      // clear existing image if there is one
      if (glyph.fImage && is_lib) {
         FT_Done_Glyph(glyph.fImage);
         glyph.fImage = nullptr;
      }
   }
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

FT_Face TTFhandle::GetFontFace() const
{
   return fFont ? fFont->face : nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Set the rotation matrix used to rotate the font outlines.

void TTFhandle::SetRotationMatrix(Float_t angle)
{
   Float_t rangle = angle * TMath::Pi() / 180.; // Angle in radian
#if defined(FREETYPE_PATCH) && \
    (FREETYPE_MAJOR == 2) && (FREETYPE_MINOR == 1) && (FREETYPE_PATCH == 2)
   Float_t sin    = TMath::Sin(rangle);
   Float_t cos    = TMath::Cos(rangle);
#else
   Float_t sin    = TMath::Sin(-rangle);
   Float_t cos    = TMath::Cos(-rangle);
#endif

   if (!fRotMatrix)
      fRotMatrix = std::make_unique<FT_Matrix>();

   fRotMatrix->xx = (FT_Fixed) (cos * (1<<16));
   fRotMatrix->xy = (FT_Fixed) (sin * (1<<16));
   fRotMatrix->yx = -fRotMatrix->xy;
   fRotMatrix->yy =  fRotMatrix->xx;
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
      Fatal("TTFhandle::SetTextFont", "Fail to create font handle for font %s", basename);
      return 1;
   }

   // font face exists and initialized
   if (fFont->face)
      return 0;

   auto lib = InitClose();
   if (!lib) {
      Error("TTFhandle::SetTextFont", "no free type library initialized");
      return 1;
   }

   // try to load font (font must be in Root.TTFontPath resource)
   const char *ttpath = gEnv->GetValue("Root.TTFontPath",
                                       TROOT::GetTTFFontDir());
   char *ttfont = gSystem->Which(ttpath, fontname, kReadPermission);

   if (!ttfont)
      return SelectFontHandle(111, TString::Format("font file %s not found in path %s", fontname, ttpath));

   std::string filename = ttfont;
   delete [] ttfont;

   if (FT_New_Face(lib, filename.c_str(), 0, &fFont->face))
      return SelectFontHandle(111, TString::Format("error loading font %s", filename.c_str()));

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
      char *ttfont = gSystem->Which(ttpath, gEnv->GetValue(fonttable[fontid][0], fonttable[fontid][1]), kReadPermission);
      if (ttfont) {
         delete [] ttfont;
         thisset = 0;
      } else {
         // try backup free font
         thisset = 1;
      }
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
   FT_Library_Version(InitClose(), &major, &minor, &patch);
}


////////////////////////////////////////////////////////////////////////////////

Bool_t TTFhandle::Init()
{
   return InitClose(1) != nullptr;
}


////////////////////////////////////////////////////////////////////////////////

Bool_t TTFhandle::GetHinting()
{
   return fgHinting;
}

////////////////////////////////////////////////////////////////////////////////

Bool_t TTFhandle::GetSmoothing()
{
   return fgSmoothing;
}

////////////////////////////////////////////////////////////////////////////////

void TTFhandle::SetHinting(Bool_t state)
{
   fgHinting = state;
}

////////////////////////////////////////////////////////////////////////////////

void TTFhandle::SetSmoothing(Bool_t state)
{
   fgSmoothing = state;
}


/** \class TTF
\ingroup BasicGraphics

Interface to the freetype 2 library.
Implements old static API.
Unitl ROOT7 just redirects to static TTFhandle instance,
then TTFhandle will be renamed into TTF class
*/

thread_local TTF gCleanupTTF; // Allows to call "Cleanup" at the end of the session
thread_local std::unique_ptr<TTFhandle> fgHandle; // static handle, destroyed automatically

////////////////////////////////////////////////////////////////////////////////
/// Cleanup TTF environment.

TTF::~TTF()
{
   TTFhandle::InitClose(-1);
}

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
   return TTFhandle::GetHinting();
}

////////////////////////////////////////////////////////////////////////////////

Bool_t TTF::GetKerning()
{
   return fgHandle ? fgHandle->GetKerning() : kFALSE;
}

////////////////////////////////////////////////////////////////////////////////

Bool_t TTF::GetSmoothing()
{
   return TTFhandle::GetSmoothing();
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
   return fgHandle ? fgHandle->GetRotMatrix() : nullptr;
}

////////////////////////////////////////////////////////////////////////////////

Int_t  TTF::GetTrailingBlanksWidth()
{
   return fgHandle ? fgHandle->GetTrailingBlanksWidth() : 0;
}

////////////////////////////////////////////////////////////////////////////////

const FT_BBox &TTF::GetBox()
{
   static FT_BBox dummy;
   return fgHandle ? fgHandle->GetBox() : dummy;
}

////////////////////////////////////////////////////////////////////////////////

TTF::TTGlyph *TTF::GetGlyphs()
{
   return fgHandle ? fgHandle->GetGlyphs() : nullptr;
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
   TTFhandle::SetHinting(state);
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
   TTFhandle::SetSmoothing(state);
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
