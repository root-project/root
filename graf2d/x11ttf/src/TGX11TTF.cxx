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


#include "TGX11TTF.h"

#ifdef R__HAS_XFT
#include "THashTable.h"
#include "TRefCnt.h"
#include <X11/Xft/Xft.h>
#endif


#include <cstdlib>

#include "TEnv.h"
#include "TError.h"
#include "TTFhandle.h"
#include "TMathBase.h"

#include "X11Drawable.h"


struct RXColor:XColor{};
struct RVisual:Visual{};
struct RXImage:XImage{};

#ifdef R__HAS_XFT

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

class X11DrawableTTF : public X11Drawable {

protected:
   Bool_t fUseTTF = kTRUE;

   ////////////////////////////////////////////////////////////////////////////////
   /// Get the background of the current window in an XImage.

   RXImage *GetBackground(Int_t x, Int_t y, UInt_t w, UInt_t h)
   {
      auto cws = (Window_t) fDrawing;
      UInt_t width;
      UInt_t height;
      Int_t xy;
      fX11.GetWindowSize(cws, xy, xy, width, height);

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

      return (RXImage *)XGetImage((Display*)fX11.fDisplay, cws, x, y, w, h, AllPlanes, ZPixmap);
   }


public:
   X11DrawableTTF(Int_t id, TGX11TTF &x11, Bool_t usettf = kTRUE) : X11Drawable(id, x11), fUseTTF(usettf) {}
   ~X11DrawableTTF() override = default;

   template <typename CharType>
   void DrawTextHelper(Int_t x, Int_t y, Float_t angle, Float_t mgn, const CharType *text, TVirtualX::ETextMode mode)
   {
      if (!fUseTTF) {
         X11Drawable::DrawText(x, y, angle, mgn, text, mode);
         return;
      }

      Window_t cws = (Window_t) fDrawing;
      UInt_t width, height;
      Int_t xy;
      fX11.GetWindowSize(cws, xy, xy, width, height);

      TTFhandle ttf;
      ttf.SetSmoothing(fX11.GetDepth() > 8);
      ttf.SetTextFont(fAttText.GetTextFont());
      ttf.SetTextSize(fAttText.GetTextSize());
      ttf.SetRotationMatrix(angle);
      ttf.PrepareString(text);
      ttf.LayoutGlyphs();
      if (ttf.ApplyAlignRotate(x, y, fAttText.GetTextAlign(), width, height))
         DrawTTFglyphs(x, y, ttf, mode);
   }

   void DrawText(Int_t x, Int_t y, Float_t angle, Float_t mgn, const char *text, TVirtualX::ETextMode mode) override
   {
      DrawTextHelper<char>(x, y, angle, mgn, text, mode);

   }
   void DrawText(Int_t x, Int_t y, Float_t angle, Float_t mgn, const wchar_t *text, TVirtualX::ETextMode mode) override
   {
      DrawTextHelper<wchar_t>(x, y, angle, mgn, text, mode);
   }

   void DrawTTFglyphs(Int_t x1, Int_t y1, TTFhandle &ttf, TVirtualX::ETextMode mode) override
   {
      Int_t w    = ttf.GetGlyphsWidth();
      Int_t h    = ttf.GetGlyphsHeight();

      Window_t cws = (Window_t) fDrawing;

      UInt_t depth = fX11.GetDepth();
      XImage *xim = XCreateImage((Display*)fX11.fDisplay, fX11.fVisual,
                                 depth, ZPixmap, 0, nullptr, w, h,
                                 depth <= 8 ? 8 : (depth <= 16 ? 16 : 32), 0);
      //bitmap_pad should be 8, 16 or 32 https://www.x.org/releases/X11R7.5/doc/man/man3/XPutPixel.3.html
      if (!xim)
         return;

      // use malloc since Xlib will use free() in XDestroyImage
      xim->data = (char *) malloc(xim->bytes_per_line * h);
      memset(xim->data, 0, xim->bytes_per_line * h);

      XGCValues values;
      XGetGCValues((Display*)fX11.fDisplay, fGClist[kGCtext], GCForeground | GCBackground, &values);

      // get the background
      if (mode == TVirtualX::kClear) {
         // if mode == kClear we need to get an image of the background
         XImage *bim = GetBackground(x1, y1, w, h);
         if (!bim) {
            ::Error("DrawTTFglyphsW", "error getting background image");
            return;
         }

         // and copy it into the text image
         Int_t xo = x1 < 0 ? -x1 : 0;
         Int_t yo = y1 < 0 ? -y1 : 0;

         for (int yp = 0; yp < (int) bim->height; yp++) {
            for (int xp = 0; xp < (int) bim->width; xp++) {
               ULong_t pixel = XGetPixel(bim, xp, yp);
               XPutPixel(xim, xo + xp, yo + yp, pixel);
            }
         }
         XDestroyImage(bim);
      } else {
         // if mode == kOpaque its simple, we just draw the background
         XAddPixel(xim, values.background);
      }

      // paint the glyphs in the XImage
      for (UInt_t nglyph = 0; nglyph < ttf.GetNumGlyphs(); nglyph++) {
         Int_t bx = 0, by = 0;
         UChar_t *buffer = nullptr;
         UInt_t width = 0, rows = 0, pitch = 0;
         if (!ttf.GetGlyphData(nglyph, bx, by, buffer, width, rows, pitch))
            continue;

         if (ttf.GetSmoothing()) {
            RXColor col[5];

            // background kClear, i.e. transparent, we take as background color
            // the average of the rgb values of all pixels covered by this character
            if (mode == TVirtualX::kClear) {
               const UInt_t ndots = TMath::Min((UInt_t) 50000, width * rows);

               std::vector<RXColor> bcol(ndots);
               if (!bcol.size()) {
                  ::Error("DrawTTFglyphs", "Allocation failure with bcol vector");
                  return;
               }
               for (unsigned y = 0, dotcnt = 0; y < rows; y++) {
                  for (unsigned x = 0; (x < width) && (dotcnt < bcol.size()); x++) {
                     bcol[dotcnt].pixel = XGetPixel(xim, bx + x, by + y);
                     bcol[dotcnt].flags = DoRed | DoGreen | DoBlue;
                     dotcnt++;
                  }
               }
               fX11.QueryColors(fX11.fColormap, bcol.data(), bcol.size());
               ULong_t r = 0, g = 0, b = 0;
               for (auto &entry : bcol) {
                  r += entry.red;
                  g += entry.green;
                  b += entry.blue;
               }
               col[0].red = (UShort_t) (r / bcol.size());
               col[0].green = (UShort_t) (g / bcol.size());
               col[0].blue = (UShort_t) (b / bcol.size());
            } else {
               // just request rgb value for background color
               col[0].pixel = values.background;
               col[0].flags = DoRed | DoGreen | DoBlue;
               fX11.QueryColors(fX11.fColormap, &col[0], 1);
            }

            // request rgb value for foreground color
            col[4].pixel = values.foreground;
            col[4].flags = DoRed | DoGreen | DoBlue;
            fX11.QueryColors(fX11.fColormap, &col[4], 1);

            // recalculate the 3 smoothing colors
            // (interpolation between fore- and background colors)
            for (int x = 3; x > 0; x--) {
               col[x].red   = (col[4].red  *x + col[0].red  *(4-x)) /4;
               col[x].green = (col[4].green*x + col[0].green*(4-x)) /4;
               col[x].blue  = (col[4].blue *x + col[0].blue *(4-x)) /4;
               if (!fX11.AllocColor(fX11.fColormap, &col[x])) {
                  ::Warning("DrawTTFglyphs", "cannot allocate smoothing color");
                  col[x].pixel = col[x+1].pixel;
               }
            }

            UChar_t *s = buffer;
            // put smoothed character, character pixmap values are an index
            // into the 5 colors used for aliasing (4 = foreground, 0 = background)
            for (unsigned y = 0; y < rows; y++) {
               for (unsigned x = 0; x < width; x++) {
                  UChar_t d = TMath::Min((UChar_t) 4, (UChar_t)((((*s++ & 0xff) + 10) * 5) / 256));
                  if (d > 0)
                     XPutPixel(xim, bx + x, by + y, col[d].pixel);
               }
            }
         } else {
            // no smoothing, just put character using foreground color
            UChar_t *row = buffer;
            for (unsigned y = 0; y < rows; y++) {
               unsigned n = 0;
               UChar_t d = 0;
               UChar_t *s = row;
               for (unsigned x = 0; x < width; x++) {
                  if (n == 0) d = *s++;
                  if (TESTBIT(d,7-n))
                     XPutPixel(xim, bx + x, by + y, values.foreground);
                  if (++n == kBitsPerByte) n = 0;
               }
               row += pitch;
            }
         }
      }

      // put the Ximage on the screen
      XPutImage((Display*)fX11.fDisplay, cws, fGClist[kGCpxmp], xim, 0, 0, x1, y1, w, h);
      XDestroyImage(xim);
   }
};


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
/// Method to create window handle capable to draw text with TTF

void TGX11TTF::CreateWindowHandle(Int_t id)
{
   fWindows.emplace(id, std::make_shared<X11DrawableTTF>(id, *this, fHasTTFonts));
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
   Bool_t res = TGX11::Init(display);

   if (res)
      TTFhandle::SetSmoothingPrefer(fDepth > 8);

   return res;
}

////////////////////////////////////////////////////////////////////////////////
/// Draw TTF glyphs on the specified window context

void TGX11TTF::DrawTTFglyphsW(WinContext_t wctxt, Int_t x1, Int_t y1, TTFhandle &ttf, ETextMode mode)
{
   auto ctxt = (X11DrawableTTF *) wctxt;
   ctxt->DrawTTFglyphs(x1, y1, ttf, mode);
}

////////////////////////////////////////////////////////////////////////////////
/// Draw text using TrueType fonts. If TrueType fonts are not available the
/// text is drawn with TGX11::DrawTextW.

void TGX11TTF::DrawTextW(WinContext_t wctxt, Int_t x, Int_t y, Float_t angle, Float_t mgn,
                         const char *text, ETextMode mode)
{
   auto ctxt = (X11DrawableTTF *) wctxt;
   ctxt->DrawText(x, y, angle, mgn, text, mode);
}

////////////////////////////////////////////////////////////////////////////////
/// Draw text using TrueType fonts. If TrueType fonts are not available the
/// text is drawn with TGX11::DrawTextW.

void TGX11TTF::DrawTextW(WinContext_t wctxt, Int_t x, Int_t y, Float_t angle, Float_t mgn,
                         const wchar_t *text, ETextMode mode)
{
   auto ctxt = (X11DrawableTTF *) wctxt;
   ctxt->DrawText(x, y, angle, mgn, text, mode);
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
