// @(#)root/win32ttf:$Name:  $:$Id: TGWin32TTF.cxx,v 1.7 2002/08/23 14:49:23 rdm Exp $
// Author: Olivier Couet     01/10/02

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGWin32TTF                                                           //
//                                                                      //
// Interface to low level Win32 . This class gives access to basic      //
// Win32 graphics via the parent class TGWin32. However, all text       //
// and font handling is done via the Freetype TrueType library.         //
// When the shared library containing this class is loaded the global   //
// gVirtualX is redirected to point to this class.                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <stdlib.h>

#include "TGWin32TTF.h"

#ifndef ROOT_GdkConstants
#include "GdkConstants.h"
#endif

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTFInit                                                              //
//                                                                      //
// Small utility class that takes care of switching the current         //
// gVirtualX to the new TGWin32TTF class as soon as the shared library    //
// containing this class is loaded.                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TTFWin32Init {
public:
   TTFWin32Init();
};

TTFWin32Init::TTFWin32Init()
{
   if (gVirtualX->IsA() == TGWin32::Class()) {
      TGWin32 *oldg = (TGWin32 *) gVirtualX;
      gVirtualX = new TGWin32TTF(*oldg);
      delete oldg;
   }
}

static TTFWin32Init gTTFWin32Init;

ClassImp(TGWin32TTF)

//______________________________________________________________________________
TGWin32TTF::TGWin32TTF(const TGWin32 &org) : TGWin32(org)
{
   // Create copy of TGWin32 but now use TrueType fonts.

   SetName("Win32TTF");
   SetTitle("ROOT interface to Win32 with TrueType fonts");

   if (!TTF::IsInitialized()) TTF::Init();

   if (fDepth > 8) {
      TTF::SetSmoothing(kTRUE);
   } else {
      TTF::SetSmoothing(kFALSE);
   }

   fHasTTFonts = kTRUE;
}

//______________________________________________________________________________
TGWin32TTF::~TGWin32TTF()
{
}

//______________________________________________________________________________
void TGWin32TTF::Align(void)
{
   // Compute alignment variables. The alignment is done on the horizontal string
   // then the rotation is applied on the alignment variables.
   // SetRotation and LayoutGlyphs should have been called before.

   EAlign align = (EAlign) fTextAlign;

   // vertical alignment
   if (align == kTLeft || align == kTCenter || align == kTRight) {
      fAlign.y = TTF::GetAscent();
   } else if (align == kMLeft || align == kMCenter || align == kMRight) {
      fAlign.y = TTF::GetAscent()/2;
   } else {
      fAlign.y = 0;
   }
   // horizontal alignment
   if (align == kTRight || align == kMRight || align == kBRight) {
      fAlign.x = TTF::GetWidth();
   } else if (align == kTCenter || align == kMCenter || align == kBCenter) {
      fAlign.x = TTF::GetWidth()/2;
   } else {
      fAlign.x = 0;
   }

   FT_Vector_Transform(&fAlign, TTF::GetRotMatrix());
   fAlign.x = fAlign.x >> 6;
   fAlign.y = fAlign.y >> 6;
}

//______________________________________________________________________________
void TGWin32TTF::DrawImage(FT_Bitmap *source, ULong_t fore, ULong_t back,
                         GdkImage *xim, Int_t bx, Int_t by)
{
   // Draw FT_Bitmap bitmap to xim image at position bx,by using specified
   // foreground color.

   UChar_t d = 0, *s = source->buffer;

   if (TTF::GetSmoothing()) {

      static GdkColor col[5];
      GdkColor *bcol = 0, *bc;
      Int_t    x, y;

      // background kClear, i.e. transparent, we take as background color
      // the average of the rgb values of all pixels covered by this character
      if (back == (ULong_t) -1 && (UInt_t)source->width) {
         ULong_t r, g, b;
         Int_t   dots, dotcnt;
         const Int_t maxdots = 50000;

         dots = Int_t(source->width * source->rows);
         dots = dots > maxdots ? maxdots : dots;
         bcol = new GdkColor[dots];
         if (!bcol) return;
         bc = bcol;
         dotcnt = 0;
         for (y = 0; y < (int) source->rows; y++) {
            for (x = 0; x < (int) source->width; x++, bc++) {
///               bc->pixel = XGetPixel(xim, bx + x, by - c->TTF::GetAscent() + y);
               bc->pixel = GetPixel((Drawable_t)xim, bx + x, by + y);
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
         if (back != (ULong_t) -1) {
            col[3].pixel = back;
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
               PutPixel((Drawable_t)xim, bx + x, by + y, p);
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
               PutPixel((Drawable_t)xim, bx + x, by + y, fore);
            if (++n == (int) kBitsPerByte) n = 0;
         }
         row += source->pitch;
      }
   }
}

//______________________________________________________________________________
void TGWin32TTF::DrawText(Int_t x, Int_t y, Float_t angle, Float_t mgn,
                        const char *text, ETextMode mode)
{
   // Draw text using TrueType fonts. If TrueType fonts are not available the
   // text is drawn with TGWin32::DrawText.

   if (!fHasTTFonts) {
      TGWin32::DrawText(x, y, angle, mgn, text, mode);
   } else {
      if (!TTF::IsInitialized()) TTF::Init();
      TTF::SetRotationMatrix(angle);
      TTF::PrepareString(text);
      TTF::LayoutGlyphs();
      Align();
      RenderString(x, y, mode);
   }

}

//______________________________________________________________________________
GdkImage *TGWin32TTF::GetBackground(Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Get the background of the current window in an XImage.

   XWindow_t *cws = TGWin32::GetCurrentWindow();
   EnterCriticalSection(flpCriticalSection);

   if (x < 0) {
      w += x;
      x  = 0;
   }
   if (y < 0) {
      h += y;
      y  = 0;
   }

   if (x+w > cws->width)  w = cws->width - x;
   if (y+h > cws->height) h = cws->height - y;

   fThreadP.Drawable = cws->drawing;
   fThreadP.x = x;
   fThreadP.y = y;
   fThreadP.w = w;
   fThreadP.h = h;
   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_IMAGE_GET, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   GdkImage *image = (GdkImage *)fThreadP.pRet;
   LeaveCriticalSection(flpCriticalSection);
   return image;
}

//______________________________________________________________________________
Bool_t TGWin32TTF::IsVisible(Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Test if there is really something to render

   XWindow_t *cws = TGWin32::GetCurrentWindow();

   // If w or h is 0, very likely the string is only blank characters
   if ((int)w == 0 || (int)h == 0)  return kFALSE;

   // If string falls outside window, there is probably no need to draw it.
   if (x + (int)w <= 0 || x >= (int)cws->width)  return kFALSE;
   if (y + (int)h <= 0 || y >= (int)cws->height) return kFALSE;

   return kTRUE;
}

//______________________________________________________________________________
void TGWin32TTF::RenderString(Int_t x, Int_t y, ETextMode mode)
{
   // Perform the string rendering in the pad.
   // LayoutGlyphs should have been called before.
   EnterCriticalSection(flpCriticalSection);
   TTGlyph* glyph = TTF::GetGlyphs();

   // compute the size and position of the XImage that will contain the text
   Int_t Xoff = 0; if (TTF::GetBox().xMin < 0) Xoff = -TTF::GetBox().xMin;
   Int_t Yoff = 0; if (TTF::GetBox().yMin < 0) Yoff = -TTF::GetBox().yMin;
   Int_t w    = TTF::GetBox().xMax + Xoff;
   Int_t h    = TTF::GetBox().yMax + Yoff;
   Int_t x1   = x-Xoff-fAlign.x;
   Int_t y1   = y+Yoff+fAlign.y-h;

   if (!IsVisible(x1, y1, w, h)) {
       LeaveCriticalSection(flpCriticalSection);
       return;
   }

   // create the XImage that will contain the text
   UInt_t depth = fDepth;
   GdkImage *xim  = 0;

   fThreadP.w = w;
   fThreadP.h = h;
   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_IMAGE_NEW, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   xim = (GdkImage *)fThreadP.pRet;

   // use malloc since Xlib will use free() in XDestroyImage
//   xim->data = (char *) malloc(xim->bytes_per_line * h);
//   memset(xim->data, 0, xim->bytes_per_line * h);

   ULong_t   bg;
//   XGCValues values;
//   XGetGCValues(fDisplay, *GetGC(3), GCForeground | GCBackground, &values);

   fThreadP.GC = GetGC(3);
   PostThreadMessage(fIDThread, WIN32_GDK_GC_GET_VALUES, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   // get the background
   if (mode == kClear) {
      // if mode == kClear we need to get an image of the background
      GdkImage *bim = GetBackground(x1, y1, w, h);
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
            ULong_t pixel = GetPixel((Drawable_t)bim, xp, yp);
            PutPixel((Drawable_t)xim, xo+xp, yo+yp, pixel);
         }
      }
      fThreadP.pParam = bim;
      PostThreadMessage(fIDThread, WIN32_GDK_IMAGE_UNREF, 0, 0L);  
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      bg = (ULong_t) -1;
   } else {
      // if mode == kOpaque its simple, we just draw the background
//      XAddPixel(xim, fThreadP.gcvals.background.pixel);
      bg = fThreadP.gcvals.background.pixel;
   }

   // paint the glyphs in the XImage
   glyph = TTF::GetGlyphs();
   for (int n = 0; n < TTF::GetNumGlyphs(); n++, glyph++) {
      if (FT_Glyph_To_Bitmap(&glyph->fImage,
                             TTF::GetSmoothing() ? ft_render_mode_normal
                                              : ft_render_mode_mono,
                             0, 1 )) continue;
      FT_BitmapGlyph bitmap = (FT_BitmapGlyph)glyph->fImage;
      FT_Bitmap*     source = &bitmap->bitmap;
      Int_t          bx, by;

      bx = bitmap->left+Xoff;
      by = h - bitmap->top-Yoff;
      DrawImage(source, fThreadP.gcvals.foreground.pixel, bg, xim, bx, by);
   }

   // put the Ximage on the screen
   XWindow_t *cws = TGWin32::GetCurrentWindow();
   GdkGC *gc = GetGC(6);      // gGCpxmp
   fThreadP.Drawable = cws->drawing;
   fThreadP.GC = gc; 
   fThreadP.pParam = xim;
   fThreadP.x = 0;
   fThreadP.y = 0;
   fThreadP.x1 = x1;
   fThreadP.y1 = y1;
   fThreadP.w = w;
   fThreadP.h = h;
   PostThreadMessage(fIDThread, WIN32_GDK_DRAW_IMAGE, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   fThreadP.pParam = xim;
   PostThreadMessage(fIDThread, WIN32_GDK_IMAGE_UNREF, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32TTF::SetTextFont(Font_t fontnumber)
{
   // Set specified font.

   fTextFont = fontnumber;
   if (!fHasTTFonts) {
      TGWin32::SetTextFont(fontnumber);
   } else {
      TTF::SetTextFont(fontnumber);
   }
}

//______________________________________________________________________________
Int_t TGWin32TTF::SetTextFont(char *fontname, ETextSetMode mode)
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
      return TGWin32::SetTextFont(fontname, mode);
   } else {
      return TTF::SetTextFont(fontname);
   }

   // set initial text size
   SetTextSize(fTextSize);
}

//______________________________________________________________________________
void TGWin32TTF::SetTextSize(Float_t textsize)
{
   // Set current text size.

   fTextSize = textsize;
   if (!fHasTTFonts) {
      TGWin32::SetTextSize(textsize);

   } else {
      TTF::SetTextSize(textsize);
   }
}
