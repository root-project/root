// @(#)root/x11ttf:$Id$
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

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGX11TTF                                                             //
//                                                                      //
// Interface to low level X11 (Xlib). This class gives access to basic  //
// X11 graphics via the parent class TGX11. However, all text and font  //
// handling is done via the Freetype TrueType library. When the         //
// shared library containing this class is loaded the global gVirtualX  //
// is redirected to point to this class.                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <stdlib.h>

#include "TGX11TTF.h"
#include "TClass.h"
#include "TEnv.h"

#ifdef R__HAS_XFT

#include "THashTable.h"
#include "TRefCnt.h"
#include <X11/Xft/Xft.h>

/////////////////////////  xft font data //////////////////////////////////////
class TXftFontData : public TNamed, public TRefCnt {
public:
   XFontStruct   *fFontStruct;   // fontstruct
   XftFont       *fXftFont;      // xft font

   TXftFontData(FontStruct_t font, XftFont *xftfont, const char *name) :
      TNamed(name, ""), TRefCnt(), fXftFont(xftfont)
   {
      SetRefCount(1);
      fFontStruct = (XFontStruct*)font;
   }

   ~TXftFontData()
   {
      if (fFontStruct) ((TGX11*)gVirtualX)->DeleteFont((FontStruct_t)fFontStruct);
      if (fXftFont) XftFontClose((Display*)gVirtualX->GetDisplay(), fXftFont);
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

   TXftFontData *FindByStruct(FontStruct_t font)
   {
      TIter next(fList);
      TXftFontData *d = 0;

      while ((d = (TXftFontData*) next())) {
         if (d->fFontStruct == (XFontStruct*)font) {
            return d;
         }
      }
      return 0;
   }

   TXftFontData *FindByHandle(FontH_t id)
   {
      TIter next(fList);
      TXftFontData *d = 0;

      while ((d = (TXftFontData*) next())) {
         if (d->fFontStruct->fid == id) {
            return d;
         }
      }
      return 0;
   }

   void AddFont(TXftFontData *data)
   {
      fList->Add(data);
   }

   void FreeFont(TXftFontData *data)
   {
      if (data->RemoveReference() > 0)  return;
      fList->Remove(data);
      delete data;
   }
};
#endif  // R__HAS_XFT

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTFX11Init                                                           //
//                                                                      //
// Small utility class that takes care of switching the current         //
// gVirtualX to the new TGX11TTF class as soon as the shared library    //
// containing this class is loaded.                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TTFX11Init {
public:
   TTFX11Init() { TGX11TTF::Activate(); }
};
static TTFX11Init gTTFX11Init;


ClassImp(TGX11TTF)

//______________________________________________________________________________
TGX11TTF::TGX11TTF(const TGX11 &org) : TGX11(org)
{
   // Create copy of TGX11 but now use TrueType fonts.

   SetName("X11TTF");
   SetTitle("ROOT interface to X11 with TrueType fonts");

   if (!TTF::fgInit) TTF::Init();

   fHasTTFonts = kTRUE;
   fAlign.x = 0;
   fAlign.y = 0;

#ifdef R__HAS_XFT
   fXftFontHash = 0;
   if (gEnv->GetValue("X11.UseXft", 0)) {
      fXftFontHash = new TXftFontHash();
   }
#endif
}

//______________________________________________________________________________
void TGX11TTF::Activate()
{
   // Static method setting TGX11TTF as the acting gVirtualX.

   if (gVirtualX && dynamic_cast<TGX11*>(gVirtualX)) {
      TGX11 *oldg = (TGX11 *) gVirtualX;
      gVirtualX = new TGX11TTF(*oldg);
      delete oldg;
   }
}

//______________________________________________________________________________
Bool_t TGX11TTF::Init(void *display)
{
   // Initialize X11 system. Returns kFALSE in case of failure.

   Bool_t r = TGX11::Init(display);

   if (fDepth > 8) {
      TTF::SetSmoothing(kTRUE);
   } else {
      TTF::SetSmoothing(kFALSE);
   }

   return r;
}

//______________________________________________________________________________
void TGX11TTF::Align(void)
{
   // Compute alignment variables. The alignment is done on the horizontal string
   // then the rotation is applied on the alignment variables.
   // SetRotation and LayoutGlyphs should have been called before.

   EAlign align = (EAlign) fTextAlign;

   // vertical alignment
   if (align == kTLeft || align == kTCenter || align == kTRight) {
      fAlign.y = TTF::fgAscent;
   } else if (align == kMLeft || align == kMCenter || align == kMRight) {
      fAlign.y = TTF::fgAscent/2;
   } else {
      fAlign.y = 0;
   }

   // horizontal alignment
   if (align == kTRight || align == kMRight || align == kBRight) {
      fAlign.x = TTF::fgWidth;
   } else if (align == kTCenter || align == kMCenter || align == kBCenter) {
      fAlign.x = TTF::fgWidth/2;
   } else {
      fAlign.x = 0;
   }

   FT_Vector_Transform(&fAlign, TTF::fgRotMatrix);
   fAlign.x = fAlign.x >> 6;
   fAlign.y = fAlign.y >> 6;
}

//______________________________________________________________________________
void TGX11TTF::DrawImage(FT_Bitmap *source, ULong_t fore, ULong_t back,
                         XImage *xim, Int_t bx, Int_t by)
{
   // Draw FT_Bitmap bitmap to xim image at position bx,by using specified
   // foreground color.

   UChar_t d = 0, *s = source->buffer;

   if (TTF::fgSmoothing) {

      static XColor col[5];
      XColor  *bcol = 0, *bc;
      Int_t    x, y;

      // background kClear, i.e. transparent, we take as background color
      // the average of the rgb values of all pixels covered by this character
      if (back == (ULong_t) -1 && (UInt_t)source->width) {
         ULong_t r, g, b;
         Int_t   dots, dotcnt;
         const Int_t maxdots = 50000;

         dots = Int_t(source->width * source->rows);
         dots = dots > maxdots ? maxdots : dots;
         bcol = new XColor[dots];
         if (!bcol) return;
         bc = bcol;
         dotcnt = 0;
         for (y = 0; y < (int) source->rows; y++) {
            for (x = 0; x < (int) source->width; x++, bc++) {
///               bc->pixel = XGetPixel(xim, bx + x, by - c->TTF::fgAscent + y);
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
      // recalculate the 3 smooting colors (interpolation between fore-
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

         // interpolate between fore and backgound colors
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

//______________________________________________________________________________
void TGX11TTF::DrawText(Int_t x, Int_t y, Float_t angle, Float_t mgn,
                        const char *text, ETextMode mode)
{
   // Draw text using TrueType fonts. If TrueType fonts are not available the
   // text is drawn with TGX11::DrawText.

   if (!fHasTTFonts) {
      TGX11::DrawText(x, y, angle, mgn, text, mode);
   } else {
      if (!TTF::fgInit) TTF::Init();
      TTF::SetRotationMatrix(angle);
      TTF::PrepareString(text);
      TTF::LayoutGlyphs();
      Align();
      RenderString(x, y, mode);
   }
}

//______________________________________________________________________________
XImage *TGX11TTF::GetBackground(Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Get the background of the current window in an XImage.

   Window_t cws = GetCurrentWindow();
   UInt_t width;
   UInt_t height;
   Int_t xy;
   gVirtualX->GetWindowSize(cws, xy, xy, width, height);

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

   return XGetImage(fDisplay, cws, x, y, w, h, AllPlanes, ZPixmap);
}

//______________________________________________________________________________
Bool_t TGX11TTF::IsVisible(Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Test if there is really something to render.

   Window_t cws = GetCurrentWindow();
   UInt_t width;
   UInt_t height;
   Int_t xy;
   gVirtualX->GetWindowSize(cws, xy, xy, width, height);

   // If w or h is 0, very likely the string is only blank characters
   if ((int)w == 0 || (int)h == 0)  return kFALSE;

   // If string falls outside window, there is probably no need to draw it.
   if (x + (int)w <= 0 || x >= (int)width)  return kFALSE;
   if (y + (int)h <= 0 || y >= (int)height) return kFALSE;

   return kTRUE;
}

//______________________________________________________________________________
void TGX11TTF::RenderString(Int_t x, Int_t y, ETextMode mode)
{
   // Perform the string rendering in the pad.
   // LayoutGlyphs should have been called before.

   TTGlyph* glyph = TTF::fgGlyphs;

   // compute the size and position of the XImage that will contain the text
   Int_t Xoff = 0; if (TTF::GetBox().xMin < 0) Xoff = -TTF::GetBox().xMin;
   Int_t Yoff = 0; if (TTF::GetBox().yMin < 0) Yoff = -TTF::GetBox().yMin;
   Int_t w    = TTF::GetBox().xMax + Xoff;
   Int_t h    = TTF::GetBox().yMax + Yoff;
   Int_t x1   = x-Xoff-fAlign.x;
   Int_t y1   = y+Yoff+fAlign.y-h;

   if (!IsVisible(x1, y1, w, h)) return;

   // create the XImage that will contain the text
   UInt_t depth = fDepth;
   XImage *xim  = 0;
   xim = XCreateImage(fDisplay, fVisual,
                      depth, ZPixmap, 0, 0, w, h,
                      depth == 24 ? 32 : (depth==15?16:depth), 0);

   // use malloc since Xlib will use free() in XDestroyImage
   xim->data = (char *) malloc(xim->bytes_per_line * h);
   memset(xim->data, 0, xim->bytes_per_line * h);

   ULong_t   bg;
   XGCValues values;
   XGetGCValues(fDisplay, *GetGC(3), GCForeground | GCBackground, &values);

   // get the background
   if (mode == kClear) {
      // if mode == kClear we need to get an image of the background
      XImage *bim = GetBackground(x1, y1, w, h);
      if (!bim) {
         Error("DrawText", "error getting background image");
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
   glyph = TTF::fgGlyphs;
   for (int n = 0; n < TTF::fgNumGlyphs; n++, glyph++) {
      if (FT_Glyph_To_Bitmap(&glyph->fImage,
                             TTF::fgSmoothing ? ft_render_mode_normal
                                              : ft_render_mode_mono,
                             0, 1 )) continue;
      FT_BitmapGlyph bitmap = (FT_BitmapGlyph)glyph->fImage;
      FT_Bitmap*     source = &bitmap->bitmap;
      Int_t          bx, by;

      bx = bitmap->left+Xoff;
      by = h - bitmap->top-Yoff;
      DrawImage(source, values.foreground, bg, xim, bx, by);
   }

   // put the Ximage on the screen
   Window_t cws = GetCurrentWindow();
   GC *gc = GetGC(6);      // gGCpxmp
   XPutImage(fDisplay, cws, *gc, xim, 0, 0, x1, y1, w, h);
   XDestroyImage(xim);
}

//______________________________________________________________________________
void TGX11TTF::SetTextFont(Font_t fontnumber)
{
   // Set specified font.

   fTextFont = fontnumber;
   if (!fHasTTFonts) {
      TGX11::SetTextFont(fontnumber);
   } else {
      TTF::SetTextFont(fontnumber);
   }
}

//______________________________________________________________________________
Int_t TGX11TTF::SetTextFont(char *fontname, ETextSetMode mode)
{
   // Set text font to specified name.
   // mode       : loading flag
   // mode=0     : search if the font exist (kCheck)
   // mode=1     : search the font and load it if it exists (kLoad)
   // font       : font name
   //
   // Set text font to specified name. This function returns 0 if
   // the specified font is found, 1 if not.

   if (!fHasTTFonts) {
      return TGX11::SetTextFont(fontname, mode);
   } else {
      return TTF::SetTextFont(fontname);
   }
}

//______________________________________________________________________________
void TGX11TTF::SetTextSize(Float_t textsize)
{
   // Set current text size.

   fTextSize = textsize;
   if (!fHasTTFonts) {
      TGX11::SetTextSize(textsize);
   } else {
      TTF::SetTextSize(textsize);
   }
}

#ifdef R__HAS_XFT

///////////////////////////// Xft font methods /////////////////////////////////
//______________________________________________________________________________
FontStruct_t TGX11TTF::LoadQueryFont(const char *font_name)
{
   // Parses an XLFD name and opens a font.

   if (!fXftFontHash) {
      return TGX11::LoadQueryFont(font_name);
   }

   TXftFontData *data = fXftFontHash->FindByName(font_name);

   // already loaded
   if (data) {
      data->AddReference();
      return (FontStruct_t)data->fFontStruct;
   }

   // load both X11 and Xft fonts
   FontStruct_t font = TGX11::LoadQueryFont(font_name);

   if (!font) {
      return font;
   }

   XftFont *xftfont = XftFontOpenXlfd(fDisplay, fScreenNumber, font_name);

   data = new TXftFontData(font, xftfont, font_name);
   fXftFontHash->AddFont(data);

   return font;
}

//______________________________________________________________________________
void TGX11TTF::DeleteFont(FontStruct_t fs)
{
   // Explicitely delete font structure obtained with LoadQueryFont().

   if (!fXftFontHash) {
      TGX11::DeleteFont(fs);
      return;
   }

   TXftFontData *data = fXftFontHash->FindByStruct(fs);

   if (!data) {
      TGX11::DeleteFont(fs);
      return;
   }

   fXftFontHash->FreeFont(data);
}

//______________________________________________________________________________
Int_t TGX11TTF::TextWidth(FontStruct_t font, const char *s, Int_t len)
{
   // Return lenght of string in pixels. Size depends on font

   if (!fXftFontHash) {
      return TGX11::TextWidth(font, s, len);
   }

   TXftFontData *data = fXftFontHash->FindByStruct(font);

   if (!data) {
      return TGX11::TextWidth(font, s, len);
   }

   XftFont *xftfont = data->fXftFont;

   if (!xftfont) {
      return TGX11::TextWidth(font, s, len);
   }

   XGlyphInfo glyph_info;
   XftTextExtents8(fDisplay, xftfont, (XftChar8 *)s, len, &glyph_info);

   return glyph_info.xOff;
}

//______________________________________________________________________________
void TGX11TTF::GetFontProperties(FontStruct_t font, Int_t &max_ascent, Int_t &max_descent)
{
   //  Return some font properties

   if (!fXftFontHash) {
      TGX11::GetFontProperties(font, max_ascent, max_descent);
      return;
   }

   TXftFontData *data = fXftFontHash->FindByStruct(font);

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

//______________________________________________________________________________
void TGX11TTF::DrawString(Drawable_t xwindow, GContext_t gc, Int_t x, Int_t y,
                          const char *text, Int_t len)
{
   // Draw text string

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

   if (!text || (len < 1) || !strlen(text)) {
      return;
   }

   if (!fXftFontHash) {
      TGX11::DrawString(xwindow, gc, x, y, text, len);
      return;
   }

   GCValues_t gval;
   gval.fMask = kGCForeground | kGCBackground | kGCFont;  // retrieve GC values
   GetGCValues(gc, gval);

   TXftFontData *data = fXftFontHash->FindByHandle(gval.fFont);

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
   XGetGeometry(fDisplay, (Drawable)xwindow, &droot, &dx, &dy,
                &width, &height, &bwidth, &depth);

   if (depth <= 1) {
      TGX11::DrawString(xwindow, gc, x, y, text, len);
      return;
   }

   memset(&xcolor, 0, sizeof(xcolor));
   xcolor.pixel = gval.fForeground;

   XQueryColor(fDisplay, fColormap, &xcolor);

   // create  XftDraw
   xftdraw = XftDrawCreate(fDisplay, (Drawable)xwindow, fVisual, fColormap);

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
