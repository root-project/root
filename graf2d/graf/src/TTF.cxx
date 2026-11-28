// @(#)root/graf:$Id$
// Author: Olivier Couet     01/10/02

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TTF
\ingroup BasicGraphics

Interface to the freetype 2 library.
*/

#  include <ft2build.h>
#  include FT_FREETYPE_H
#  include FT_GLYPH_H
#include "TROOT.h"
#include "TTF.h"
#include "TSystem.h"
#include "TEnv.h"
#include "TMath.h"
#include "TError.h"

#include <fontconfig/fontconfig.h>

// to scale fonts to the same size as the old TT version
const Float_t kScale = 0.93376068;

TTF gCleanupTTF; // Allows to call "Cleanup" at the end of the session

Bool_t         TTF::fgInit           = kFALSE;
Bool_t         TTF::fgSmoothing      = kTRUE;
Bool_t         TTF::fgKerning        = kTRUE;
Bool_t         TTF::fgHinting        = kFALSE;
Int_t          TTF::fgTBlankW        = 0;
Int_t          TTF::fgWidth          = 0;
Int_t          TTF::fgAscent         = 0;
Int_t          TTF::fgCurFontIdx     = -1;
Int_t          TTF::fgFontCount      = 0;
Int_t          TTF::fgNumGlyphs      = 0;
char          *TTF::fgFontName[kTTMaxFonts];
Int_t          TTF::fgFontIdx[kTTMaxFonts];
Int_t          TTF::fgFontIta[kTTMaxFonts];
FT_Matrix     *TTF::fgRotMatrix      = nullptr;
FT_Library     TTF::fgLibrary;
FT_BBox        TTF::fgCBox;
FT_Face        TTF::fgFace[kTTMaxFonts];
FT_CharMap     TTF::fgCharMap[kTTMaxFonts];
TTF::TTGlyph   TTF::fgGlyphs[kMaxGlyphs];


////////////////////////////////////////////////////////////////////////////////
/// Cleanup TTF environment.

TTF::~TTF()
{
   Cleanup();
}

////////////////////////////////////////////////////////////////////////////////
/// Initialise the TrueType fonts interface.

void TTF::Init()
{
   fgInit = kTRUE;

   // initialize FTF library
   if (FT_Init_FreeType(&fgLibrary)) {
      Error("TTF::Init", "error initializing FreeType");
      return;
   }

   // Add root's font directory
   const char *ttpath = gEnv->GetValue("Root.TTFontPath",
                                       TROOT::GetTTFFontDir());
   FcConfigAppFontAddDir (nullptr, (const FcChar8*)ttpath);

   // load default font (arialbd)
   SetTextFont(62);
}

////////////////////////////////////////////////////////////////////////////////
/// Cleanup. Is called by the gCleanupTTF destructor.

void TTF::Cleanup()
{
   if (!fgInit) return;

   for (int i = 0; i < fgFontCount; i++) {
      delete [] fgFontName[i];
      FT_Done_Face(fgFace[i]);
   }
   if (fgRotMatrix) delete fgRotMatrix;
   FT_Done_FreeType(fgLibrary);

   fgInit = kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Map char to unicode. Returns 0 in case no mapping exists.

Short_t TTF::CharToUnicode(UInt_t code)
{
   if (!fgCharMap[fgCurFontIdx]) {
      UShort_t i, platform, encoding;
      FT_CharMap  charmap;

      if (!fgFace[fgCurFontIdx]) return 0;
      Int_t n = fgFace[fgCurFontIdx]->num_charmaps;
      for (i = 0; i < n; i++) {
         if (!fgFace[fgCurFontIdx]) continue;
         charmap  = fgFace[fgCurFontIdx]->charmaps[i];
         platform = charmap->platform_id;
         encoding = charmap->encoding_id;
         if ((platform == 3 && encoding == 1 &&
              (fgFontIta[fgCurFontIdx] & 2) == 0) ||
             (platform == 0 && encoding == 0) ||
             (platform == 7 && encoding == 2 &&
              (fgFontIta[fgCurFontIdx] & 2) != 0) ||
             (platform == 0 && encoding == 3 &&
              (fgFontIta[fgCurFontIdx] & 2) != 0) ||
             (platform == 1 && encoding == 0 &&
              (fgFontIta[fgCurFontIdx] & 2) != 0))
         {
            fgCharMap[fgCurFontIdx] = charmap;
            if (FT_Set_Charmap(fgFace[fgCurFontIdx], fgCharMap[fgCurFontIdx]))
                Error("TTF::CharToUnicode", "error in FT_Set_CharMap");
            return FT_Get_Char_Index(fgFace[fgCurFontIdx], (FT_ULong)code);
         }
      }
   }
   return FT_Get_Char_Index(fgFace[fgCurFontIdx], (FT_ULong)code);
}

////////////////////////////////////////////////////////////////////////////////
/// Compute the trailing blanks width. It is use to compute the text width in GetTextExtent
/// `n` is the number of trailing blanks in a string.

void TTF::ComputeTrailingBlanksWidth(Int_t n)
{
   fgTBlankW = 0;
   if (n) {
      FT_Face face = fgFace[fgCurFontIdx];
      char space = ' ';
      FT_UInt load_flags = FT_LOAD_DEFAULT;
      if (!fgHinting) load_flags |= FT_LOAD_NO_HINTING;
      FT_Load_Char(face, space, load_flags);

      FT_GlyphSlot slot      = face->glyph;
      FT_Pos advance_x       = slot->advance.x;
      Int_t advance_x_pixels = advance_x >> 6;

      fgTBlankW = advance_x_pixels * n;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Get width (w) and height (h) when text is horizontal.

void TTF::GetTextExtent(UInt_t &w, UInt_t &h, char *text)
{
   if (!fgInit) Init();

   SetRotationMatrix(0);
   PrepareString(text);
   LayoutGlyphs();
   Int_t Xoff = 0; if (fgCBox.xMin < 0) Xoff = -fgCBox.xMin;
   Int_t Yoff = 0; if (fgCBox.yMin < 0) Yoff = -fgCBox.yMin;
   w = fgCBox.xMax + Xoff + GetTrailingBlanksWidth();
   h = fgCBox.yMax + Yoff;
}

////////////////////////////////////////////////////////////////////////////////
/// Get advance (a) when text is horizontal.

void TTF::GetTextAdvance(UInt_t &a, char *text)
{
   if (!fgInit) Init();

   SetRotationMatrix(0);
   PrepareString(text);
   LayoutGlyphs();
   a = GetWidth()>>6;
}

////////////////////////////////////////////////////////////////////////////////
/// Get width (w) and height (h) when text is horizontal.

void TTF::GetTextExtent(UInt_t &w, UInt_t &h, wchar_t *text)
{
   if (!fgInit) Init();

   SetRotationMatrix(0);
   PrepareString(text);
   LayoutGlyphs();
   Int_t Xoff = 0; if (fgCBox.xMin < 0) Xoff = -fgCBox.xMin;
   Int_t Yoff = 0; if (fgCBox.yMin < 0) Yoff = -fgCBox.yMin;
   w = fgCBox.xMax + Xoff + GetTrailingBlanksWidth();
   h = fgCBox.yMax + Yoff;
}

////////////////////////////////////////////////////////////////////////////////
/// Compute the glyphs positions, fgAscent and fgWidth (needed for alignment).
/// Perform the Glyphs transformation.
/// Compute the string control box.
/// If required take the "kerning" into account.
/// SetRotation and PrepareString should have been called before.

void TTF::LayoutGlyphs()
{
   TTGlyph*  glyph = fgGlyphs;
   FT_Vector origin;
   FT_UInt   load_flags;
   FT_UInt   prev_index = 0;

   fgAscent = 0;
   fgWidth  = 0;

   load_flags = FT_LOAD_DEFAULT;
   if (!fgHinting) load_flags |= FT_LOAD_NO_HINTING;

   fgCBox.xMin = fgCBox.yMin =  32000;
   fgCBox.xMax = fgCBox.yMax = -32000;

   for (int n = 0; n < fgNumGlyphs; n++, glyph++) {

      // compute glyph origin
      if (fgKerning) {
         if (prev_index) {
            FT_Vector  kern;
            FT_Get_Kerning(fgFace[fgCurFontIdx], prev_index, glyph->fIndex,
                           fgHinting ? ft_kerning_default : ft_kerning_unfitted,
                           &kern);
            fgWidth += kern.x;
         }
         prev_index = glyph->fIndex;
      }

      origin.x = fgWidth;
      origin.y = 0;

      // clear existing image if there is one
      if (glyph->fImage) {
         FT_Done_Glyph(glyph->fImage);
         glyph->fImage = nullptr;
      }

      // load the glyph image (in its native format)
      if (FT_Load_Glyph(fgFace[fgCurFontIdx], glyph->fIndex, load_flags))
         continue;

      // extract the glyph image
      if (FT_Get_Glyph (fgFace[fgCurFontIdx]->glyph, &glyph->fImage))
         continue;

      glyph->fPos = origin;
      fgWidth    += fgFace[fgCurFontIdx]->glyph->advance.x;
      fgAscent    = TMath::Max((Int_t)(fgFace[fgCurFontIdx]->glyph->metrics.horiBearingY), fgAscent);

      // transform the glyphs
      FT_Vector_Transform(&glyph->fPos, fgRotMatrix);
      if (FT_Glyph_Transform(glyph->fImage, fgRotMatrix, &glyph->fPos))
         continue;

      // compute the string control box
      FT_BBox  bbox;
      FT_Glyph_Get_CBox(glyph->fImage, ft_glyph_bbox_pixels, &bbox);
      if (bbox.xMin < fgCBox.xMin) fgCBox.xMin = bbox.xMin;
      if (bbox.yMin < fgCBox.yMin) fgCBox.yMin = bbox.yMin;
      if (bbox.xMax > fgCBox.xMax) fgCBox.xMax = bbox.xMax;
      if (bbox.yMax > fgCBox.yMax) fgCBox.yMax = bbox.yMax;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Put the characters in "string" in the "glyphs" array.

void TTF::PrepareString(const char *string)
{
   const unsigned char *p = (const unsigned char*) string;
   TTGlyph *glyph = fgGlyphs;
   UInt_t index;       // Unicode value
   Int_t NbTBlank = 0; // number of trailing blanks

   fgNumGlyphs = 0;
   while (*p) {
      index = CharToUnicode((FT_ULong)*p);
      if (index != 0) {
         glyph->fIndex = index;
         glyph++;
         fgNumGlyphs++;
      }
      if (*p == ' ') {
         NbTBlank++;
      } else {
         NbTBlank = 0;
      }
      if (fgNumGlyphs >= kMaxGlyphs) break;
      p++;
   }

   ComputeTrailingBlanksWidth(NbTBlank);
}

////////////////////////////////////////////////////////////////////////////////
/// Put the characters in "string" in the "glyphs" array.

void TTF::PrepareString(const wchar_t *string)
{
   const wchar_t *p = string;
   TTGlyph *glyph = fgGlyphs;
   UInt_t index;       // Unicode value
   Int_t NbTBlank = 0; // number of trailing blanks

   fgNumGlyphs = 0;
   while (*p) {
      index = FT_Get_Char_Index(fgFace[fgCurFontIdx], (FT_ULong)*p);
      if (index != 0) {
         glyph->fIndex = index;
         glyph++;
         fgNumGlyphs++;
      }
      if (*p == ' ') {
         NbTBlank++;
      } else {
         NbTBlank = 0;
      }
      if (fgNumGlyphs >= kMaxGlyphs) break;
      p++;
   }

   ComputeTrailingBlanksWidth(NbTBlank);
}

////////////////////////////////////////////////////////////////////////////////
/// Set hinting flag.

void TTF::SetHinting(Bool_t state)
{
   fgHinting = state;
}

////////////////////////////////////////////////////////////////////////////////
/// Set kerning flag.

void TTF::SetKerning(Bool_t state)
{
   fgKerning = state;
}

////////////////////////////////////////////////////////////////////////////////
/// Set the rotation matrix used to rotate the font outlines.

void TTF::SetRotationMatrix(Float_t angle)
{
   Float_t rangle = Float_t(angle * TMath::Pi() / 180.); // Angle in radian
#if defined(FREETYPE_PATCH) && \
    (FREETYPE_MAJOR == 2) && (FREETYPE_MINOR == 1) && (FREETYPE_PATCH == 2)
   Float_t sin    = TMath::Sin(rangle);
   Float_t cos    = TMath::Cos(rangle);
#else
   Float_t sin    = TMath::Sin(-rangle);
   Float_t cos    = TMath::Cos(-rangle);
#endif

   if (!fgRotMatrix) fgRotMatrix = new FT_Matrix;

   fgRotMatrix->xx = (FT_Fixed) (cos * (1<<16));
   fgRotMatrix->xy = (FT_Fixed) (sin * (1<<16));
   fgRotMatrix->yx = -fgRotMatrix->xy;
   fgRotMatrix->yy =  fgRotMatrix->xx;
}

////////////////////////////////////////////////////////////////////////////////
/// Set smoothing (anti-aliasing) flag.

void TTF::SetSmoothing(Bool_t state)
{
   fgSmoothing = state;
}

////////////////////////////////////////////////////////////////////////////////
/// Set text font to specified name.
///  - font       : font name
///  - italic     : the fonts should be slanted. Used for symbol font.
///
/// Set text font to specified name. This function returns 0 if
/// the specified font is found, 1 if not.

Int_t TTF::SetTextFont(const char *fontname, Int_t italic)
{
   if (!fgInit) Init();

   if (!fontname || !fontname[0]) {
      Warning("TTF::SetTextFont",
              "no font name specified, using default font %s", fgFontName[0]);
      fgCurFontIdx = 0;
      return 0;
   }
   const char *basename = gSystem->BaseName(fontname);

   char *ttfont = nullptr;
   int ttindex = 0;

   FcPattern *pat = nullptr, *match;
   FcResult result;

   if (strcmp(basename, "timesi.ttf") == 0 ||
       strcmp(basename, "FreeSerifItalic.otf") == 0) {
      pat = FcNameParse ((const FcChar8*) "freeserif:italic");
   }
   else if (strcmp(basename, "timesbd.ttf") == 0 ||
            strcmp(basename, "FreeSerifBold.otf") == 0) {
      pat = FcNameParse ((const FcChar8*) "freeserif:bold");
   }
   else if (strcmp(basename, "timesbi.ttf") == 0 ||
            strcmp(basename, "FreeSerifBoldItalic.otf") == 0) {
      pat = FcNameParse ((const FcChar8*) "freeserif:bold:italic");
   }
   else if (strcmp(basename, "arial.ttf") == 0 ||
            strcmp(basename, "FreeSans.otf") == 0) {
      pat = FcNameParse ((const FcChar8*) "freesans");
   }
   else if (strcmp(basename, "ariali.ttf") == 0 ||
            strcmp(basename, "FreeSansOblique.otf") == 0) {
      pat = FcNameParse ((const FcChar8*) "freesans:italic");
   }
   else if (strcmp(basename, "arialbd.ttf") == 0 ||
            strcmp(basename, "FreeSansBold.otf") == 0) {
      pat = FcNameParse ((const FcChar8*) "freesans:bold");
   }
   else if (strcmp(basename, "arialbi.ttf") == 0 ||
            strcmp(basename, "FreeSansBoldOblique.otf") == 0) {
      pat = FcNameParse ((const FcChar8*) "freesans:bold:italic");
   }
   else if (strcmp(basename, "cour.ttf") == 0 ||
            strcmp(basename, "FreeMono.otf") == 0) {
      pat = FcNameParse ((const FcChar8*) "freemono");
   }
   else if (strcmp(basename, "couri.ttf") == 0 ||
            strcmp(basename, "FreeMonoOblique.otf") == 0) {
      pat = FcNameParse ((const FcChar8*) "freemono:italic");
   }
   else if (strcmp(basename, "courbd.ttf") == 0 ||
            strcmp(basename, "FreeMonoBold.otf") == 0) {
      pat = FcNameParse ((const FcChar8*) "freemono:bold");
   }
   else if (strcmp(basename, "courbi.ttf") == 0 ||
            strcmp(basename, "FreeMonoBoldOblique.otf") == 0) {
      pat = FcNameParse ((const FcChar8*) "freemono:bold:italic");
   }
   else if (strcmp(basename, "symbol.ttf") == 0) {
      pat = FcNameParse ((const FcChar8*) "standardsymbolsps");
      italic &= 2;
   }
   else if (strcmp(basename, "times.ttf") == 0 ||
            strcmp(basename, "FreeSerif.otf") == 0) {
      pat = FcNameParse ((const FcChar8*) "freeserif");
   }
   else if (strcmp(basename, "wingding.ttf") == 0) {
      pat = FcNameParse ((const FcChar8*) "dingbats");
      italic &= 2;
   }
   else if (strcmp(basename, "STIXGeneral.otf") == 0) {
      pat = FcNameParse ((const FcChar8*) "stixgeneral");
   }
   else if (strcmp(basename, "STIXGeneralItalic.otf") == 0) {
      pat = FcNameParse ((const FcChar8*) "stixgeneral:italic");
   }
   else if (strcmp(basename, "STIXGeneralBol.otf") == 0) {
      pat = FcNameParse ((const FcChar8*) "stixgeneral:bold");
   }
   else if (strcmp(basename, "STIXGeneralBolIta.otf") == 0) {
      pat = FcNameParse ((const FcChar8*) "stixgeneral:bold:italic");
   }
   else if (strcmp(basename, "STIXSiz1Sym.otf") == 0) {
      pat = FcNameParse ((const FcChar8*) "stixsize1");
   }
   else if (strcmp(basename, "STIXSiz1SymBol.otf") == 0) {
      pat = FcNameParse ((const FcChar8*) "stixsize1:bold");
   }
   else if (strcmp(basename, "STIXSiz2Sym.otf") == 0) {
      pat = FcNameParse ((const FcChar8*) "stixsize2");
   }
   else if (strcmp(basename, "STIXSiz2SymBol.otf") == 0) {
      pat = FcNameParse ((const FcChar8*) "stixsize2:bold");
   }
   else if (strcmp(basename, "STIXSiz3Sym.otf") == 0) {
      pat = FcNameParse ((const FcChar8*) "stixsize3");
   }
   else if (strcmp(basename, "STIXSiz3SymBol.otf") == 0) {
      pat = FcNameParse ((const FcChar8*) "stixsize3:bold");
   }
   else if (strcmp(basename, "STIXSiz4Sym.otf") == 0) {
      pat = FcNameParse ((const FcChar8*) "stixsize4");
   }
   else if (strcmp(basename, "STIXSiz4SymBol.otf") == 0) {
      pat = FcNameParse ((const FcChar8*) "stixsize4:bold");
   }
   else if (strcmp(basename, "STIXSiz5Sym.otf") == 0) {
      pat = FcNameParse ((const FcChar8*) "stixsize5");
   }
   else if (strcmp(basename, "DroidSansFallback.ttf") == 0) {
      pat = FcNameParse ((const FcChar8*) "droidsansfallback:charset=4e00 0410");
   }
   else if (strcmp(basename, "BlackChancery.ttf") == 0) {
      pat = FcNameParse ((const FcChar8*) "urwchanceryl");
   }
   else if (gSystem->AccessPathName(fontname, kReadPermission) != 0) {
      // Font name is not a file - usae as pattern
      pat = FcNameParse ((const FcChar8*) fontname);
   }

   if (pat) {
      FcConfigSubstitute (nullptr, pat, FcMatchPattern);
      FcDefaultSubstitute (pat);
      match = FcFontMatch (nullptr, pat, &result);
      if (match) {
         const char *ttfnt;
         FcPatternGetString (match, FC_FILE, 0, (FcChar8**) &ttfnt);
         ttfont = StrDup(ttfnt);
         FcPatternGetInteger (match, FC_INDEX, 0, &ttindex);
         FcPatternDestroy (match);
      }
      FcPatternDestroy (pat);
   }

   if (!ttfont) ttfont = StrDup(fontname);

   basename = gSystem->BaseName(ttfont);

   // check if font is in cache
   int i;
   for (i = 0; i < fgFontCount; i++) {
      if (!strcmp(fgFontName[i], basename) &&
          (fgFontIdx[i] == ttindex) &&
          (fgFontIta[i] == italic)) {
         fgCurFontIdx = i;
         delete [] ttfont;
         return 0;
      }
   }

   // enough space in cache to load font?
   if (fgFontCount >= kTTMaxFonts) {
      Error("TTF::SetTextFont", "too many fonts opened (increase kTTMaxFont = %d)",
            kTTMaxFonts);
      Warning("TTF::SetTextFont", "using default font %s", fgFontName[0]);
      fgCurFontIdx = 0;    // use font 0 (default font, set in ctor)
      delete [] ttfont;
      return 0;
   }

   FT_Face  tface = (FT_Face) 0;

   if (FT_New_Face(fgLibrary, ttfont, ttindex, &tface)) {
      Error("TTF::SetTextFont", "error loading font %s", ttfont);
      delete [] ttfont;
      if (tface) FT_Done_Face(tface);
      if (fgFontCount) {
         Warning("TTF::SetTextFont", "using default font %s", fgFontName[0]);
         fgCurFontIdx = 0;
         return 0;
      } else {
         return 1;
      }
   }

   fgFontName[fgFontCount] = StrDup(basename);
   fgFontIdx[fgFontCount]  = ttindex;
   fgFontIta[fgFontCount]  = italic;
   fgCurFontIdx            = fgFontCount;
   fgFace[fgCurFontIdx]    = tface;
   fgCharMap[fgCurFontIdx] = (FT_CharMap) 0;
   fgFontCount++;

   if ((italic & 1) != 0) {
      FT_Matrix slantMat;
      slantMat.xx = (1 << 16);
      slantMat.xy = ((1 << 16) >> 2);
      slantMat.yx = 0;
      slantMat.yy = (1 << 16);
      FT_Set_Transform( fgFace[fgCurFontIdx], &slantMat, nullptr );
   }

   delete [] ttfont;

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

void TTF::SetTextFont(Font_t fontnumber)
{
   static const char *fonttable[] = {
      "freesans:bold",
      "freeserif:italic",
      "freeserif:bold",
      "freeserif:bold:italic",
      "freesans",
      "freesans:italic",
      "freesans:bold",
      "freesans:bold:italic",
      "freemono",
      "freemono:italic",
      "freemono:bold",
      "freemono:bold:italic",
      "standardsymbolsps",
      "freeserif",
      "dingbats",
      "standardsymbolsps",
      "stixgeneral",
      "stixgeneral:italic",
      "stixgeneral:bold",
      "stixgeneral:bold:italic",
      "stixsize1",
      "stixsize1:bold",
      "stixsize2",
      "stixsize2:bold",
      "stixsize3",
      "stixsize3:bold",
      "stixsize4",
      "stixsize4:bold",
      "stixsize5",
      "droidsansfallback:charset=4e00 0410",
      "droidsansfallback:charset=4e00 0410",
      "droidsansfallback:charset=4e00 0410"
   };

   int fontid = fontnumber / 10;
   if (fontid < 0 || fontid > 31) fontid = 0;

   Int_t italic = 0;
   if (fontid==12) italic = 2;
   if (fontid==14) italic = 2;
   if (fontid==15) italic = 3;

   SetTextFont(fonttable[fontid], italic);
}

////////////////////////////////////////////////////////////////////////////////
/// Set current text size.

void TTF::SetTextSize(Float_t textsize)
{
   if (!fgInit) Init();
   if (textsize < 0) return;

   if (fgCurFontIdx < 0 || fgFontCount <= fgCurFontIdx) {
      Error("TTF::SetTextSize", "current font index out of bounds");
      fgCurFontIdx = 0;
      return;
   }

   Int_t tsize = (Int_t)(textsize*kScale+0.5) << 6;
   FT_Error err = FT_Set_Char_Size(fgFace[fgCurFontIdx], tsize, tsize, 72, 72);
   if (err)
      Error("TTF::SetTextSize", "error in FT_Set_Char_Size: 0x%x (input size %f, calc. size 0x%x)", err, textsize,
            tsize);
}

////////////////////////////////////////////////////////////////////////////////

void TTF::Version(Int_t &major, Int_t &minor, Int_t &patch)
{
   FT_Library_Version(fgLibrary, &major, &minor, &patch);
}

////////////////////////////////////////////////////////////////////////////////

Bool_t TTF::GetHinting()
{
   return fgHinting;
}

////////////////////////////////////////////////////////////////////////////////

Bool_t TTF::GetKerning()
{
   return fgKerning;
}

////////////////////////////////////////////////////////////////////////////////

Bool_t TTF::GetSmoothing()
{
   return fgSmoothing;
}

////////////////////////////////////////////////////////////////////////////////

Bool_t TTF::IsInitialized()
{
   return fgInit;
}

////////////////////////////////////////////////////////////////////////////////

Int_t  TTF::GetWidth()
{
   return fgWidth;
}

////////////////////////////////////////////////////////////////////////////////

Int_t  TTF::GetAscent()
{
   return fgAscent;
}

////////////////////////////////////////////////////////////////////////////////

Int_t  TTF::GetNumGlyphs()
{
   return fgNumGlyphs;
}

////////////////////////////////////////////////////////////////////////////////

FT_Matrix *TTF::GetRotMatrix()
{
   return fgRotMatrix;
}

////////////////////////////////////////////////////////////////////////////////

Int_t  TTF::GetTrailingBlanksWidth()
{
   return fgTBlankW;
}

////////////////////////////////////////////////////////////////////////////////

const FT_BBox &TTF::GetBox()
{
   return fgCBox;
}

////////////////////////////////////////////////////////////////////////////////

TTF::TTGlyph *TTF::GetGlyphs()
{
    return fgGlyphs;
}
