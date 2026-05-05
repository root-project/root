// @(#)root/x11ttf:$Id: 80028b538e60290371c1c5d73728f78b1c32f09a $
// Author: Valeriy Onuchin (Xft support)  02/10/07
// Author: Olivier Couet     01/10/02
// Author: Fons Rademakers   21/11/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TGX11TTF
\ingroup x11

Interface to low level X11 (Xlib). This class gives access to basic
X11 graphics via the parent class TGX11. However, all text and font
handling is done via the Freetype TrueType library. When the
shared library containing this class is loaded the global gVirtualX
is redirected to point to this class.
*/

#include <cstdlib>

#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_GLYPH_H
#include "TGX11TTF.h"
#include "TEnv.h"
#include "TTF.h"
#include "TMathBase.h"


#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xatom.h>
#include <X11/cursorfont.h>
#include <X11/keysym.h>
#include <X11/xpm.h>

struct RXColor:XColor{};
struct RVisual:Visual{};
struct RXImage:XImage{};

#ifdef R__HAS_XFT

#include "THashTable.h"
#include "TRefCnt.h"
#include <X11/Xft/Xft.h>

/////////////////////////  xft font data //////////////////////////////////////
class TXftFontData : public TNamed, public TRefCnt {
public:
   GContext_t     fGC;           // graphics context
   XftFont       *fXftFont;      // xft font

   TXftFontData(GContext_t gc, XftFont *xftfont, const char *name) :
      TNamed(name, ""), TRefCnt(), fXftFont(xftfont)
   {
      SetRefCount(1);
      fGC = gc;
   }

   void MapGCFont(GContext_t gc, FontStruct_t font)
   {
      fGC = gc; fXftFont = (XftFont *)font;
   }

   ~TXftFontData() override
   {
      if (References() == 1) {
         if (fXftFont) XftFontClose((Display*)gVirtualX->GetDisplay(), fXftFont);
      }
   }
};

/////////////////// hash table //////////////////////////////////////////////
class TXftFontHash {
public:
   THashTable  *fList;  // hash table

   TXftFontHash() { fList = new THashTable(50); }

   TXftFontData *FindByName(const char *name)
   {
      return (TXftFontData*)fList->FindObject(name);
   }

   TXftFontData *FindByFont(FontStruct_t font)
   {
      TIter next(fList);

      while (auto d = (TXftFontData*) next()) {
         if (d->fXftFont == (XftFont *)font)
            return d;
      }
      return nullptr;
   }

   TXftFontData *FindByGC(GContext_t gc)
   {
      TIter next(fList);

      while (auto d = (TXftFontData*) next()) {
         if (d->fGC == gc)
            return d;
      }
      return nullptr;
   }

   void AddFont(TXftFontData *data)
   {
      // Loop over all existing TXftFontData, if we already have one with the same
      // font data, set the reference counter of this one beyond 1 so it does
      // delete the font pointer
      TIter next(fList);

      while (auto d = (TXftFontData*) next()) {
         if (d->fXftFont == data->fXftFont)
            data->AddReference();
      }

      fList->Add(data);
   }

   void FreeFont(TXftFontData *data)
   {
      fList->Remove(data);
      delete data;
   }
};

Bool_t TGX11TTF::gXftInit = kFALSE;

#endif  // R__HAS_XFT

/** \class TTFX11Init
\ingroup GraphicsBackends

Small utility class that takes care of switching the current
gVirtualX to the new TGX11TTF class as soon as the shared library
containing this class is loaded.
*/

class TTFX11Init {
public:
   TTFX11Init() { TGX11TTF::Activate(); }
};
static TTFX11Init gTTFX11Init;



////////////////////////////////////////////////////////////////////////////////
/// Create copy of TGX11 but now use TrueType fonts.

TGX11TTF::TGX11TTF(TGX11 &&org) : TGX11(std::move(org))
{
   SetName("X11TTF");
   SetTitle("ROOT interface to X11 with TrueType fonts");

   fHasTTFonts = TTFhandle::Init();
   fHasXft = kFALSE;

#ifdef R__HAS_XFT
   fXftFontHash = nullptr;
#endif
}

////////////////////////////////////////////////////////////////////////////////
/// Static method setting TGX11TTF as the acting gVirtualX.

void TGX11TTF::Activate()
{
   if (auto oldg = dynamic_cast<TGX11*>(gVirtualX)) {
      gVirtualX = new TGX11TTF(std::move(*oldg));
      delete oldg;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize X11 system. Returns kFALSE in case of failure.

Bool_t TGX11TTF::Init(void *display)
{
#ifdef R__HAS_XFT
   fXftFontHash = nullptr;
   XFontStruct *fs = nullptr;
   if (display) fs = XLoadQueryFont((Display *)display, "-*-helvetica-*-r-*-*-14-*-*-*-*-*-*-*");
   if (!fs) gEnv->SetValue("X11.UseXft", 1);
   if (display && fs) XFreeFont((Display *)display, fs);
   if (gEnv->GetValue("X11.UseXft", 0)) {
      fHasXft = kTRUE;
      fXftFontHash = new TXftFontHash();
   }
#endif
   Bool_t r = TGX11::Init(display);

   TTFhandle::SetSmoothing(fDepth > 8);

   return r;
}

////////////////////////////////////////////////////////////////////////////////
/// Draw FT_Bitmap bitmap to xim image at position bx,by using specified
/// foreground color.

void TGX11TTF::DrawFTGlyph(void *_source, ULong_t fore, ULong_t back,
                           RXImage *xim, Int_t bx, Int_t by)
{
   auto source = (FT_Bitmap *) _source;

   UChar_t d = 0, *s = source->buffer;

   if (TTFhandle::GetSmoothing()) {

      static RXColor col[5];
      RXColor  *bcol = nullptr;
      XColor  *bc;
      Int_t    x, y;

      // background kClear, i.e. transparent, we take as background color
      // the average of the rgb values of all pixels covered by this character
      if (back == (ULong_t) -1 && (UInt_t)source->width) {
         ULong_t r, g, b;
         Int_t   dots, dotcnt;
         const Int_t maxdots = 50000;

         dots = Int_t(source->width * source->rows);
         dots = dots > maxdots ? maxdots : dots;
         bcol = new RXColor[dots];
         if (!bcol) return;
         bc = bcol;
         dotcnt = 0;
         for (y = 0; y < (int) source->rows; y++) {
            for (x = 0; x < (int) source->width; x++, bc++) {
///               bc->pixel = XGetPixel(xim, bx + x, by - c->TTF::GetAscent() + y);
               bc->pixel = XGetPixel(xim, bx + x, by + y);
               bc->flags = DoRed | DoGreen | DoBlue;
               if (++dotcnt >= maxdots) break;
            }
         }
         QueryColors(fColormap, bcol, dots);
         r = g = b = 0;
         bc = bcol;
         dotcnt = 0;
         for (y = 0; y < (int) source->rows; y++) {
            for (x = 0; x < (int) source->width; x++, bc++) {
               r += bc->red;
               g += bc->green;
               b += bc->blue;
               if (++dotcnt >= maxdots) break;
            }
         }
         if (dots != 0) {
            r /= dots;
            g /= dots;
            b /= dots;
         }
         bc = &col[0];
         if (bc->red == r && bc->green == g && bc->blue == b)
            bc->pixel = back;
         else {
            bc->pixel = ~back;
            bc->red   = (UShort_t) r;
            bc->green = (UShort_t) g;
            bc->blue  = (UShort_t) b;
         }
      }
      delete [] bcol;

      // if fore or background have changed from previous character
      // recalculate the 3 smoothing colors (interpolation between fore-
      // and background colors)
      if (fore != col[4].pixel || back != col[0].pixel) {
         col[4].pixel = fore;
         col[4].flags = DoRed|DoGreen|DoBlue;
         if (back != (ULong_t) -1) {
            col[3].pixel = back;
            col[3].flags = DoRed | DoGreen | DoBlue;
            QueryColors(fColormap, &col[3], 2);
            col[0] = col[3];
         } else {
            QueryColors(fColormap, &col[4], 1);
         }

         // interpolate between fore and background colors
         for (x = 3; x > 0; x--) {
            col[x].red   = (col[4].red  *x + col[0].red  *(4-x)) /4;
            col[x].green = (col[4].green*x + col[0].green*(4-x)) /4;
            col[x].blue  = (col[4].blue *x + col[0].blue *(4-x)) /4;
            if (!AllocColor(fColormap, &col[x])) {
               Warning("DrawImage", "cannot allocate smoothing color");
               col[x].pixel = col[x+1].pixel;
            }
         }
      }

      // put smoothed character, character pixmap values are an index
      // into the 5 colors used for aliasing (4 = foreground, 0 = background)
      for (y = 0; y < (int) source->rows; y++) {
         for (x = 0; x < (int) source->width; x++) {
            d = *s++ & 0xff;
            d = ((d + 10) * 5) / 256;
            if (d > 4) d = 4;
            if (d && x < (int) source->width) {
               ULong_t p = col[d].pixel;
               XPutPixel(xim, bx + x, by + y, p);
            }
         }
      }
   } else {
      // no smoothing, just put character using foreground color
      UChar_t* row=s;
      for (int y = 0; y < (int) source->rows; y++) {
         int n = 0;
         s = row;
         for (int x = 0; x < (int) source->width; x++) {
            if (n == 0) d = *s++;
            if (TESTBIT(d,7-n))
               XPutPixel(xim, bx + x, by + y, fore);
            if (++n == (int) kBitsPerByte) n = 0;
         }
         row += source->pitch;
      }
   }
}

template<class CharType>
void TGX11TTF::DrawTextHelper(WinContext_t wctxt, Int_t x, Int_t y, Float_t angle, Float_t mgn,
                              const CharType *text, ETextMode mode)
{
   if (!fHasTTFonts) {
      TGX11::DrawTextW(wctxt, x, y, angle, mgn, text, mode);
      return;
   }

   if (!wctxt)
      return;

   auto &att = GetTextAttW(wctxt);
   auto align = GetTextAlignW(wctxt);

   TTFhandle ttf;
   ttf.SetTextFont(att.GetTextFont());
   ttf.SetTextSize(att.GetTextSize());
   ttf.SetRotationMatrix(angle);
   ttf.PrepareString(text);
   ttf.LayoutGlyphs();

   FT_Vector   align_vect;                 ///< alignment vector
   // vertical alignment
   if (align == kTLeft || align == kTCenter || align == kTRight) {
      align_vect.y = ttf.GetAscent();
   } else if (align == kMLeft || align == kMCenter || align == kMRight) {
      align_vect.y = ttf.GetAscent() / 2;
   } else {
      align_vect.y = 0;
   }

   // horizontal alignment
   if (align == kTRight || align == kMRight || align == kBRight) {
      align_vect.x = ttf.GetWidth();
   } else if (align == kTCenter || align == kMCenter || align == kBCenter) {
      align_vect.x = ttf.GetWidth() / 2;
   } else {
      align_vect.x = 0;
   }

   FT_Vector_Transform(&align_vect, ttf.GetRotMatrix());
   align_vect.x = align_vect.x >> 6;
   align_vect.y = align_vect.y >> 6;

   Int_t Xoff = TMath::Max(0, (Int_t) -ttf.GetBox().xMin);
   Int_t Yoff = TMath::Max(0, (Int_t) -ttf.GetBox().yMin);
   Int_t w    = ttf.GetBox().xMax + Xoff;
   Int_t h    = ttf.GetBox().yMax + Yoff;
   // If w or h is 0, very likely the string is only blank characters
   if (w <= 0 || h <= 0)
      return;

   Int_t x1   = x - Xoff - align_vect.x;
   Int_t y1   = y + Yoff + align_vect.y - h;

   Window_t cws = GetWindow(wctxt);
   UInt_t width, height;
   Int_t xy;
   GetWindowSize(cws, xy, xy, width, height);

   // If string falls outside window, there is probably no need to draw it.
   if (x + w <= 0 || x >= (Int_t)width || y + h <= 0 || y >= (Int_t)height)
      return;

   // If w or h are much larger than the window size, there is probably no need
   // to draw it. Moreover a to large text size may produce a Seg Fault in
   // malloc in DrawTextW.
   if (((UInt_t) w > 10 * width) || ((UInt_t) h > 10 * height))
      return;

   // create the XImage that will contain the text
   UInt_t depth = fDepth;
   XImage *xim = XCreateImage((Display*)fDisplay, fVisual,
                               depth, ZPixmap, 0, nullptr, w, h,
                               depth <= 8 ? 8 : (depth <= 16 ? 16 : 32), 0);
   //bitmap_pad should be 8, 16 or 32 https://www.x.org/releases/X11R7.5/doc/man/man3/XPutPixel.3.html
   if (!xim)
      return;

   // use malloc since Xlib will use free() in XDestroyImage
   xim->data = (char *) malloc(xim->bytes_per_line * h);
   memset(xim->data, 0, xim->bytes_per_line * h);

   ULong_t   bg;
   XGCValues values;
   auto gc = (GC *) GetGCW(wctxt, 3);
   if (!gc) {
      Error("DrawTextW", "error getting Graphics Context");
      return;
   }
   XGetGCValues((Display*)fDisplay, *gc, GCForeground | GCBackground, &values);

   // get the background
   if (mode == kClear) {
      // if mode == kClear we need to get an image of the background
      XImage *bim = GetBackground(wctxt, x1, y1, w, h);
      if (!bim) {
         Error("DrawTextW", "error getting background image");
         return;
      }

      // and copy it into the text image
      Int_t xo = 0, yo = 0;
      if (x1 < 0) xo = -x1;
      if (y1 < 0) yo = -y1;

      for (int yp = 0; yp < (int) bim->height; yp++) {
         for (int xp = 0; xp < (int) bim->width; xp++) {
            ULong_t pixel = XGetPixel(bim, xp, yp);
            XPutPixel(xim, xo+xp, yo+yp, pixel);
         }
      }
      XDestroyImage(bim);
      bg = (ULong_t) -1;
   } else {
      // if mode == kOpaque its simple, we just draw the background
      XAddPixel(xim, values.background);
      bg = values.background;
   }

   // paint the glyphs in the XImage
   for (UInt_t n = 0; n < ttf.GetNumGlyphs(); n++) {
      if (auto bitmap = ttf.GetGlyphBitmap(n)) {
         Int_t bx = bitmap->left + Xoff;
         Int_t by = h - bitmap->top - Yoff;
         DrawFTGlyph(&bitmap->bitmap, values.foreground, bg, (RXImage *)xim, bx, by);
      }
   }

   // put the Ximage on the screen
   gc = (GC *) GetGCW(wctxt, 6);
   if (gc)
      XPutImage((Display*)fDisplay, cws, *gc, xim, 0, 0, x1, y1, w, h);
   XDestroyImage(xim);
}

////////////////////////////////////////////////////////////////////////////////
/// Draw text using TrueType fonts. If TrueType fonts are not available the
/// text is drawn with TGX11::DrawTextW.

void TGX11TTF::DrawTextW(WinContext_t wctxt, Int_t x, Int_t y, Float_t angle, Float_t mgn,
                         const char *text, ETextMode mode)
{
   DrawTextHelper(wctxt, x, y, angle, mgn, text, mode);
}

////////////////////////////////////////////////////////////////////////////////
/// Draw text using TrueType fonts. If TrueType fonts are not available the
/// text is drawn with TGX11::DrawTextW.

void TGX11TTF::DrawTextW(WinContext_t wctxt, Int_t x, Int_t y, Float_t angle, Float_t mgn,
                         const wchar_t *text, ETextMode mode)
{
   DrawTextHelper(wctxt, x, y, angle, mgn, text, mode);
}

////////////////////////////////////////////////////////////////////////////////
/// Get the background of the current window in an XImage.

RXImage *TGX11TTF::GetBackground(WinContext_t wctxt, Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   Window_t cws = GetWindow(wctxt);
   UInt_t width;
   UInt_t height;
   Int_t xy;
   GetWindowSize(cws, xy, xy, width, height);

   if (x < 0) {
      w += x;
      x  = 0;
   }
   if (y < 0) {
      h += y;
      y  = 0;
   }

   if (x+w > width)  w = width - x;
   if (y+h > height) h = height - y;

   return (RXImage *)XGetImage((Display*)fDisplay, cws, x, y, w, h, AllPlanes, ZPixmap);
}


#ifdef R__HAS_XFT

///////////////////////////// Xft font methods /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/// Parses an XLFD name and opens a font.

FontStruct_t TGX11TTF::LoadQueryFont(const char *font_name)
{
   if (!fXftFontHash) {
      return TGX11::LoadQueryFont(font_name);
   }

   TXftFontData *data = fXftFontHash->FindByName(font_name);

   // already loaded
   if (data) {
      return (FontStruct_t)data->fXftFont;
   }

   if (!gXftInit) {
      XftInit(nullptr);
      gXftInit = kTRUE;
   }

   XftFont *xftfont = XftFontOpenXlfd((Display*)fDisplay, fScreenNumber, font_name);

   data = new TXftFontData(0, xftfont, font_name);
   fXftFontHash->AddFont(data);

   return (FontStruct_t)xftfont;
}

////////////////////////////////////////////////////////////////////////////////
/// Explicitly delete font structure obtained with LoadQueryFont().

void TGX11TTF::DeleteFont(FontStruct_t fs)
{
   if (!fXftFontHash) {
      TGX11::DeleteFont(fs);
      return;
   }

   TXftFontData *data = fXftFontHash->FindByFont(fs);

   if (data)
      fXftFontHash->FreeFont(data);
}

////////////////////////////////////////////////////////////////////////////////
/// Explicitly delete a graphics context.

void TGX11TTF::DeleteGC(GContext_t gc)
{
   if (!fXftFontHash) {
      TGX11::DeleteGC(gc);
      return;
   }

   TXftFontData *gcdata = fXftFontHash->FindByGC(gc);
   if (gcdata) fXftFontHash->FreeFont(gcdata);
   TGX11::DeleteGC(gc);
}

////////////////////////////////////////////////////////////////////////////////
/// Return handle to font described by font structure.

FontH_t TGX11TTF::GetFontHandle(FontStruct_t fs)
{
   if (!fXftFontHash) {
      return TGX11::GetFontHandle(fs);
   }

   return (FontH_t)fs;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the font associated with the graphics context gc

FontStruct_t TGX11TTF::GetGCFont(GContext_t gc)
{
   if (!fXftFontHash) {
      return 0;
   }

   TXftFontData *data = fXftFontHash->FindByGC(gc);

   // no XftFont data
   if (!data) return 0;

   return (FontStruct_t)data->fXftFont;
}

////////////////////////////////////////////////////////////////////////////////
/// Map the XftFont with the Graphics Context using it.

void TGX11TTF::MapGCFont(GContext_t gc, FontStruct_t font)
{
   if (!fXftFontHash)
      return;

   TXftFontData *gcdata = fXftFontHash->FindByGC(gc);
   TXftFontData *fontdata = fXftFontHash->FindByFont(font);

   if (gcdata) { // && (gcdata->fXftFont == 0)) {
      gcdata->fXftFont = (XftFont *)font;
   }
   else if (fontdata) {
      TXftFontData *data = new TXftFontData(gc, (XftFont *)font, fontdata->GetName());
      fXftFontHash->AddFont(data);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Return length of string in pixels. Size depends on font

Int_t TGX11TTF::TextWidth(FontStruct_t font, const char *s, Int_t len)
{
   if (!fXftFontHash) {
      return TGX11::TextWidth(font, s, len);
   }

   TXftFontData *data = fXftFontHash->FindByFont(font);

   if (!data) return 0;

   XftFont *xftfont = data->fXftFont;

   if (xftfont) {
      XGlyphInfo glyph_info;
      XftTextExtents8((Display *)fDisplay, xftfont, (XftChar8 *)s, len, &glyph_info);
      return glyph_info.xOff;
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
///  Return some font properties

void TGX11TTF::GetFontProperties(FontStruct_t font, Int_t &max_ascent, Int_t &max_descent)
{
   if (!fXftFontHash) {
      TGX11::GetFontProperties(font, max_ascent, max_descent);
      return;
   }

   TXftFontData *data = fXftFontHash->FindByFont(font);

   if (!data) {
      TGX11::GetFontProperties(font, max_ascent, max_descent);
      return;
   }

   XftFont *xftfont = data->fXftFont;

   if (!xftfont) {
      TGX11::GetFontProperties(font, max_ascent, max_descent);
      return;
   }

   max_ascent = xftfont->ascent;
   max_descent = xftfont->descent;
}

////////////////////////////////////////////////////////////////////////////////
/// Draw text string

void TGX11TTF::DrawString(Drawable_t xwindow, GContext_t gc, Int_t x, Int_t y,
                          const char *text, Int_t len)
{
   XftDraw  *xftdraw;
   XftColor  xftcolor;
   XColor    xcolor;
   XftFont  *xftfont;

   if (!xwindow)  {
      return;
   }

   if (!gc) {
      return;
   }

   if (!text || (len < 1) || !text[0]) {
      return;
   }

   if (!fXftFontHash) {
      TGX11::DrawString(xwindow, gc, x, y, text, len);
      return;
   }

   GCValues_t gval;
   gval.fMask = kGCForeground | kGCBackground;  // retrieve GC values
   GetGCValues(gc, gval);

   TXftFontData *data = fXftFontHash->FindByGC(gc);

   // no XftFont data
   if (!data) {
      TGX11::DrawString(xwindow, gc, x, y, text, len);
      return;
   }

   xftfont = data->fXftFont;

   // no Xft font
   if (!xftfont) {
      TGX11::DrawString(xwindow, gc, x, y, text, len);
      return;
   }

   // dummies
   Window droot;
   Int_t dx,dy;
   UInt_t bwidth, width, height, depth;

   // check if drawable is bitmap
   XGetGeometry((Display*)fDisplay, (Drawable)xwindow, &droot, &dx, &dy,
                &width, &height, &bwidth, &depth);

   if (depth <= 1) {
      TGX11::DrawString(xwindow, gc, x, y, text, len);
      return;
   }

   memset(&xcolor, 0, sizeof(xcolor));
   xcolor.pixel = gval.fForeground;

   XQueryColor((Display*)fDisplay, fColormap, &xcolor);

   // create  XftDraw
   xftdraw = XftDrawCreate((Display*)fDisplay, (Drawable)xwindow, fVisual, fColormap);

   if (!xftdraw) {
      //Warning("could not create an XftDraw");
      TGX11::DrawString(xwindow, gc, x, y, text, len);
      return;
   }

   xftcolor.color.red = xcolor.red;
   xftcolor.color.green = xcolor.green;
   xftcolor.color.blue = xcolor.blue;
   xftcolor.color.alpha = 0xffff;
   xftcolor.pixel = gval.fForeground;

   XftDrawString8(xftdraw, &xftcolor, xftfont, x, y, (XftChar8 *)text, len);

   // cleanup
   XftDrawDestroy(xftdraw);
}

#endif // R__HAS_XFT
