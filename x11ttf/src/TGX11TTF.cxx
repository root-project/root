// @(#)root/x11ttf:$Name$:$Id$
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
// shared library containing this class is loaded the global gVirtualX is    //
// redirected to point to this class.                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifdef HAVE_CONFIG
#include "config.h"
#endif

#include "TGX11TTF.h"
#include "TSystem.h"
#include "TEnv.h"
#include "TList.h"
#include "THashTable.h"
#include "TMath.h"


ClassImp(TGX11TTF)


// to scale TT fonts to same size as X11 fonts
const Float_t kScale = 0.7;



//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTFInit                                                              //
//                                                                      //
// Small utility class that takes care of switching the current gVirtualX    //
// to the new TGX11TTF class as soon as the shared library containing   //
// this class is loaded.                                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TTFInit {
public:
   TTFInit();
};

TTFInit::TTFInit()
{
   if (gVirtualX->IsA() == TGX11::Class()) {
      TGX11 *oldg = (TGX11 *) gVirtualX;
      gVirtualX = new TGX11TTF(*oldg);
      delete oldg;
   }
}

static TTFInit gTTFInit;


// Class (actually structure) containing character description

class TTChar : public TObject {

public:
   UInt_t      fSize;      // char size (= nint(fTextSize*kScale))
   UInt_t      fWidth;     // bitmap width
   UInt_t      fBwidth;    // bitmap width, byte boundary
   UInt_t      fHeight;    // bitmap height
   UInt_t      fCode;      // actual char code
   UInt_t      fCharlen;   // origin x axis advance width
   Float_t     fAngle;     // rotation angle in degrees
   Int_t       fAscent;    // (top of bitmap) - (origin y axis)
   Int_t       fDescent;   // (origin y axis) - (bottom of bitmap)
   Int_t       fXoff;      // (left of bitmap) - (origin x axis)
   const char *fFontName;  // font name (pointer to, not a copy)
   UChar_t    *fBitmap;    // bitmap representing character
   TObjLink   *fLink;      // keep track where in LRU list we are stored

   TTChar();
   ~TTChar();
   Bool_t  IsEqual(TObject *obj);
   ULong_t Hash();
};

TTChar::TTChar()
{
   fFontName = 0;
   fBitmap   = 0;
}

TTChar::~TTChar()
{
   delete [] fBitmap;
}

Bool_t TTChar::IsEqual(TObject *obj)
{
   TTChar *c = (TTChar *) obj;
   if (fCode == c->fCode && fSize == c->fSize && fAngle == c->fAngle &&
       !strcmp(fFontName, c->fFontName)) return kTRUE;
   return kFALSE;
}

ULong_t TTChar::Hash()
{
   return fCode ^ fSize;
}


inline Long_t TTFloor(Long_t x) { return x & -64; }
inline Long_t TTCeil(Long_t x) { return (x + 63) & -64; }


//______________________________________________________________________________
TGX11TTF::TGX11TTF(const TGX11 &org) : TGX11(org)
{
   // Create copy of TGX11 but now use TrueType fonts.

   SetName("X11TTF");
   SetTitle("ROOT interface to X11 with TrueType fonts");

   fFontCount   = 0;
   fCurFontIdx  = -1;
   fCacheCount  = 0;
   fCacheHits   = 0;
   fCacheMisses = 0;
   fRotMatrix   = 0;

   fHinting     = kTRUE;
   fSmoothing   = kFALSE;
   if (DefaultDepth(fDisplay,fScreenNumber) > 8)
      fSmoothing = kTRUE;

   TT_Error error;

   // Initialize TTF engine
   fEngine = new TT_Engine;
   if ((error = TT_Init_FreeType(fEngine))) {
      Error("TGX11TTF", "error initializing engine, code = %d", (int)error);
      return;
   }

   fHasTTFonts = kTRUE;

   // load default font (arialbd), if this fails we'll continue
   // using X11 fonts (SetTextFont will turn off fHasTTFonts)
   SetTextFont(62);

   fCharCache = new THashTable(kHashSize);
   fLRU       = new TList;
}

//______________________________________________________________________________
TGX11TTF::~TGX11TTF()
{
   // Cleanup.

   for (int i = 0; i < fFontCount; i++) {
      delete fCharMap[i];
      delete fProperties[i];
      TT_Close_Face(*fFace[i]);
      delete fFace[i];
      delete fGlyph[i];
      delete fInstance[i];
      delete [] fFontName[i];
   }

   delete fRotMatrix;

   TT_Done_FreeType(*fEngine);
   delete fEngine;

   ClearCache();
   delete fCharCache;
   delete fLRU;
}

//______________________________________________________________________________
void TGX11TTF::ClearCache()
{
   // Clear TTChar cache.

   fLRU->Clear();  // must be before the cache delete, or use option "nodelete"
   fCharCache->Delete();

   fCacheCount  = 0;
   fCacheHits   = 0;
   fCacheMisses = 0;
}

//______________________________________________________________________________
void TGX11TTF::Align(UInt_t w, UInt_t, Int_t ascent, Int_t &x, Int_t &y)
{
   // Change x,y position depending on required alignment.

   EAlign align = (EAlign) fTextAlign;
   Int_t nl = 1;   // number of lines

   if (align == kTLeft || align == kTCenter || align == kTRight)
      y += ascent;
   else if (align == kMLeft || align == kMCenter || align == kMRight)
      y = y - nl*ascent/2 + ascent;
   else if (align == kBLeft || align == kBCenter || align == kBRight)
      y = y - nl*ascent + ascent;
   else    // kNone
      y = y;

   if (align == kTLeft || align == kMLeft || align == kBLeft || align == kNone)
      x = x;
   else if (align == kTCenter || align == kMCenter || align == kBCenter)
      x -= (Int_t) w / 2;
   else
      x -= w;
}

//______________________________________________________________________________
void TGX11TTF::AlignRotated(UInt_t w, UInt_t, Int_t, Int_t yoff,
                            Int_t &x, Int_t &y)
{
   // Change x,y position or rotated text depending on required alignment.

   EAlign align = (EAlign) fTextAlign;
   Int_t nl = 1;   // number of lines

   if (align == kTLeft || align == kTCenter || align == kTRight)
      y += yoff;
   else if (align == kMLeft || align == kMCenter || align == kMRight)
      y = y - nl*yoff/2 + yoff;
   else if (align == kBLeft || align == kBCenter || align == kBRight)
      y = y - nl*yoff + yoff;
   else    // kNone
      y = y;

   if (align == kTLeft || align == kMLeft || align == kBLeft || align == kNone)
      x = x;
   else if (align == kTCenter || align == kMCenter || align == kBCenter)
      x -= (Int_t) w / 2;
   else
      x -= w;
}

//______________________________________________________________________________
void TGX11TTF::DrawText(Int_t x, Int_t y, Float_t angle, Float_t mgn,
                        const char *text, ETextMode mode)
{
   // Draw text using TrueType fonts.

#ifndef R__TTFROT
   if (!fHasTTFonts || angle != 0.) {
#else
   if (!fHasTTFonts) {
#endif
      TGX11::DrawText(x, y, angle, mgn, text, mode);
      return;
   }

   if (!text || !*text) return;

   // angles are between 0<=angle<360 degrees
   while (angle < 0.)
      angle += 360.;

   while (angle >= 360.)
      angle -= 360.;

   // FreeType rotates clockwise
   if (angle > 0)
      angle = 360 - angle;

   angle = (Float_t)TMath::Nint(angle);   // steps of 1 degree

   if (angle != 0.) {
      DrawRotatedText(x, y, angle, text, mode);
      return;
   }

   if (gDebug > 0)
      printf("TGX11TTF::DrawText: (x=%d,y=%d,ang=%f,mgn=%f,size=%f,mode=%s) %s\n",
             x, y, angle, mgn, fTextSize, mode==0 ? "kClear" : "kOpaque", text);

   // get size of string so we can create an XImage for it
   UInt_t w, h;
   Int_t  maxa;    // maximum ascent

   GetTextExtent(w, h, maxa, text);

   if (w == 0) {
      if (gDebug > 0)
         Error("DrawText", "text width is 0, something went very wrong");
      return;
   }

   Align(w, h, maxa, x, y);

   Int_t y1 = y - maxa; // top of image (while y is the text baseline)

   if (!IsVisible(x, y1, w, h)) return;

   // create image that will contain the text
   UInt_t depth = DefaultDepth(fDisplay, fScreenNumber);
   XImage *xim = XCreateImage(fDisplay, DefaultVisual(fDisplay, fScreenNumber),
                              depth, ZPixmap, 0, 0, w, h,
                              depth == 24 ? 32 : (depth==15?16:depth), 0);

   // use malloc since Xlib will use free() in XDestroyImage
   xim->data = (char *) malloc(xim->bytes_per_line * h);
   memset(xim->data, 0, xim->bytes_per_line * h);

   ULong_t   bg;
   XGCValues values;
   XGetGCValues(fDisplay, *GetGC(3), GCForeground | GCBackground, &values);

   if (mode == kClear) {
      // if mode == kClear we need to get an image of the background
      XImage *bim = GetBackground(x, y1, w, h);
      if (!bim) {
         Error("DrawText", "error getting background image");
         return;
      }

      // and copy it into the text image
      Int_t xoff = 0, yoff = 0;
      if (x  < 0) xoff = -x;
      if (y1 < 0) yoff = -y1;

      for (int yp = 0; yp < (int) bim->height; yp++) {
         for (int xp = 0; xp < (int) bim->width; xp++) {
            ULong_t pixel = XGetPixel(bim, xp, yp);
            XPutPixel(xim, xoff+xp, yoff+yp, pixel);
         }
      }
      XDestroyImage(bim);
      bg = (ULong_t) -1;
   } else {
      // if mode == kOpaque its simple, we just draw the background
      XAddPixel(xim, values.background);
      bg = values.background;
   }

   // loop over all characters and draw them in the xim image
   Int_t bx = 0;
   const char *s = text;
   while (s && *s) {
      UInt_t code = (UChar_t) *s++;
      TTChar *c = GetChar(code, TMath::Nint(fTextSize * kScale), 0.);
      if (!c) continue;
      DrawImage(c, values.foreground, bg, xim, bx, maxa);
      bx += c->fCharlen;
   }

   // put image back to pixmap on X server
   XWindow_t *cws = GetCurrentWindow();
   GC *gc = GetGC(6);      // gGCpxmp
   XPutImage(fDisplay, cws->drawing, *gc, xim, 0, 0, x, y1, w, h);

   // cleanup
   XDestroyImage(xim);
}

//______________________________________________________________________________
void TGX11TTF::DrawRotatedText(Int_t x, Int_t y, Float_t angle, const char *text,
                               ETextMode mode)
{
   // Draw rotated text using TrueType fonts.

   if (gDebug > 0)
      printf("TGX11TTF::DrawRotatedText: (x=%d,y=%d,ang=%f,size=%f,mode=%s) %s\n",
             x, y, angle, fTextSize, mode==0 ? "kClear" : "kOpaque", text);

   SetRotationMatrix(angle);

    // get size of string so we can create an XImage for it
   UInt_t w, h;
   Int_t  xoff, yoff;    // offsets from desired x and y position

   GetRotatedTextExtent(w, h, xoff, yoff, angle, text);

   printf("GetRotatedTextExtent: angle = %f, width = %d, height = %d, xoff = %d, yoff = %d\n",
          angle,w,h,xoff,yoff);

   if (w == 0) {
      if (gDebug > 0)
         Error("DrawText", "text width is 0, something went very wrong");
      return;
   }

   AlignRotated(w, h, xoff, yoff, x, y);

   Int_t y1 = y - yoff; // top of image (while y is the text baseline)
   Int_t x1 = x + xoff;

   if (!IsVisible(x1, y1, w, h)) return;

   // create image that will contain the text
   UInt_t depth = DefaultDepth(fDisplay, fScreenNumber);
   XImage *xim = XCreateImage(fDisplay, DefaultVisual(fDisplay, fScreenNumber),
                              depth, ZPixmap, 0, 0, w, h,
                              depth == 24 ? 32 : (depth==15?16:depth), 0);

   // use malloc since Xlib will use free() in XDestroyImage
   xim->data = (char *) malloc(xim->bytes_per_line * h);
   memset(xim->data, 0, xim->bytes_per_line * h);

   ULong_t   bg;
   XGCValues values;
   XGetGCValues(fDisplay, *GetGC(3), GCForeground | GCBackground, &values);

   if (mode == kClear) {
      // if mode == kClear we need to get an image of the background
      XImage *bim = GetBackground(x1, y1, w, h);
      if (!bim) {
         Error("DrawText", "error getting background image");
         return;
      }

      // and copy it into the text image
      Int_t xoff1 = 0, yoff1 = 0;
      if (x1 < 0) xoff1 = -x1;
      if (y1 < 0) yoff1 = -y1;

      for (int yp = 0; yp < (int) bim->height; yp++) {
         for (int xp = 0; xp < (int) bim->width; xp++) {
            ULong_t pixel = XGetPixel(bim, xp, yp);
            XPutPixel(xim, xoff1+xp, yoff1+yp, pixel);
         }
      }
      XDestroyImage(bim);
      bg = (ULong_t) -1;
   } else {
      // if mode == kOpaque its simple, we just draw the background
      XAddPixel(xim, values.background);
      bg = values.background;
   }

   // loop over all characters and draw them in the xim image
   Int_t bx = 0, by = 0;
   if (angle > 180.) by = h;
   const char *s = text;
   while (s && *s) {
      UInt_t code = (UChar_t) *s++;
      TTChar *c = GetChar(code, TMath::Nint(fTextSize * kScale), angle);
      if (!c) continue;
      //DrawImage(c, values.foreground, bg, xim, bx, by+c->fAscent);
      DrawImage(c, values.foreground, bg, xim, bx, by);

      TT_Pos vec_x, vec_y;
      vec_x = c->fCharlen << 6;
      vec_y = 0;
      TT_Transform_Vector(&vec_x, &vec_y, fRotMatrix);
      bx += Int_t(vec_x >> 6);
      if (angle > 180.)
         by -= Int_t(vec_y >> 6);
      else
         by += Int_t(vec_y >> 6);
   }

   // put image back to pixmap on X server
   XWindow_t *cws = GetCurrentWindow();
   GC *gc = GetGC(6);      // gGCpxmp
   XPutImage(fDisplay, cws->drawing, *gc, xim, 0, 0, x1, y1, w, h);

   // cleanup
   XDestroyImage(xim);
}

//______________________________________________________________________________
void TGX11TTF::GetTextExtent(UInt_t &w, UInt_t &h, Int_t &maxAscent, const char *text)
{
   // Get bounding box of text, returns also the maximum ascent
   // in the string. This is used to position the string (at y-maxAscent).
   // Private interface.

   w = h = 0;
   maxAscent = 0;

   const char *s = text;
   int maxa = 0, maxd = 0, extra = 0;

   // loop over all characters in the string and calculate length + height
   while (s && *s) {
      UInt_t code = (UChar_t) *s++;
      TTChar *c = GetChar(code, TMath::Nint(fTextSize * kScale), 0.);
      if (!c) continue;
      w += c->fCharlen;
      extra = c->fXoff + c->fWidth - c->fCharlen;
      maxa = TMath::Max(maxa, c->fAscent);
      maxd = TMath::Max(maxd, c->fDescent);

      if (gDebug > 1)
         printf("char %c: a = %d, d = %d, w = %d\n", code, c->fAscent, c->fDescent, c->fCharlen);
   }

   if (extra > 0) w += extra;
   h = maxa + maxd;
   maxAscent = maxa;

   if (gDebug > 0)
      printf("TGX11TTF::GetTextExtent: %d, %d, %d, %s\n", w, h, maxa, text);
}

//______________________________________________________________________________
void TGX11TTF::GetRotatedTextExtent(UInt_t &w, UInt_t &h, Int_t &xoff, Int_t &yoff,
                                    Float_t angle, const char *text)
{
   // Get bounding box of rotated text, returns also the maximum descent
   // in the string. This is used to position the string. Private interface.

   w = h = 0;
   xoff = yoff = 0;

   const char *s = text;
   int x = 0, y = 0, maxx = 0, maxy = 0, minx = 9999, miny = 9999;

   // loop over all characters in the string and calculate length + height
   while (s && *s) {
      UInt_t code = (UChar_t) *s++;
      TTChar *c = GetChar(code, TMath::Nint(fTextSize * kScale), angle);
      if (!c) continue;

      minx = TMath::Min(minx, x+c->fXoff);
      miny = TMath::Min(miny, y+c->fDescent);
      maxx = TMath::Max(maxx, int(x+c->fXoff+c->fWidth));
      maxy = TMath::Max(maxy, int(y+c->fDescent+c->fHeight));

      TT_Pos vec_x, vec_y;
      vec_x = c->fCharlen << 6;
      vec_y = 0;
      TT_Transform_Vector(&vec_x, &vec_y, fRotMatrix);
      x += Int_t(vec_x >> 6);
      y += Int_t(vec_y >> 6);

      if (gDebug > 1)
         printf("char %c: a = %d, d = %d, w = %d\n", code, c->fAscent, c->fDescent, c->fCharlen);
   }

   w = maxx - minx;
   h = maxy - miny;
   xoff = TMath::Min(minx, maxx);
   yoff = TMath::Max(miny, maxy);

   if (gDebug > 0)
      printf("TGX11TTF::GetRotatedTextExtent: %d, %d, %d, %d, %s\n", w, h, xoff, yoff, text);
}

//______________________________________________________________________________
void TGX11TTF::GetTextExtent(UInt_t &w, UInt_t &h, char *text)
{
   // Get bounding box of text.

#ifndef R__TTFROT
   if (!fHasTTFonts || fTextAngle != 0.) {
#else
   if (!fHasTTFonts) {
#endif
      TGX11::GetTextExtent(w, h, text);
      return;
   }

   Int_t maxa;
   GetTextExtent(w, h, maxa, text);
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

#ifndef R__TTFROT
   if (!fHasTTFonts || fTextAngle != 0.)
#else
   if (!fHasTTFonts)
#endif
      return TGX11::SetTextFont(fontname, mode);

   if (gDebug > 0)
      printf("SetTextFont: %s (%s)\n", fontname, mode==kLoad?"kLoad":"kCheck");

   if (!fontname || !fontname[0]) {
      Error("SetTextFont", "no font name specified");
      if (mode == kCheck) return 1;
      Warning("SetTextFont", "using default font %s", fFontName[0]);
      fCurFontIdx = 0;    // use font 0 (default font, set in ctor)
      return 0;
   }

   const char *basename = gSystem->BaseName(fontname);

   // check if font is in cache
   int i;
   for (i = 0; i < fFontCount; i++) {
      if (!strcmp(fFontName[i], basename)) {
         if (mode == kCheck) return 0;
         fCurFontIdx = i;
         return 0;
      }
   }

   // enough space in cache to load font?
   if (fFontCount >= kTTMaxFonts) {
      Error("SetTextFont", "too many fonts opened (increase kTTMaxFont = %d)",
            kTTMaxFonts);
      if (mode == kCheck) return 1;
      Warning("SetTextFont", "using default font %s", fFontName[0]);
      fCurFontIdx = 0;    // use font 0 (default font, set in ctor)
      return 0;
   }

   // try to load font (font must be in Root.TTFontPath resource)
#ifndef TTFFONTDIR
   const char *ttpath = gEnv->GetValue("Root.TTFontPath",
				       "$(ROOTSYS)/ttf/fonts");
#else
   const char *ttpath = gEnv->GetValue("Root.TTFontPath",
                                       TTFFONTDIR );
#endif

   char *ttfont = gSystem->Which(ttpath, fontname, kReadPermission);

   if (!ttfont) {
      Error("SetTextFont", "font file %s not found in path", fontname);
      if (mode == kCheck) return 1;
      if (fFontCount) {
         Warning("SetTextFont", "using default font %s", fFontName[0]);
         fCurFontIdx = 0;    // use font 0 (default font, set in ctor)
         return 0;
      } else {
         Error("SetTextFont", "switching back to X11 fonts");
         fHasTTFonts = kFALSE;
         return 1;
      }
   }

   if (mode == kCheck) {
      delete [] ttfont;
      return 0;
   }

   Int_t     res = 96;   // dpi for SVGA display
   TT_Error  error;
   TT_Face  *tface = new TT_Face;

   error = TT_Open_Face(*fEngine, ttfont, tface);
   if (error) {
      Error("SetTextFont", "error loading font %s", ttfont);
      delete [] ttfont;
      delete tface;
      if (mode == kCheck) return 1;
      if (fFontCount) {
         Warning("SetTextFont", "using default font %s", fFontName[0]);
         fCurFontIdx = 0;    // use font 0 (default font, set in ctor)
         return 0;
      } else {
         Error("SetTextFont", "switching back to X11 fonts");
         fHasTTFonts = kFALSE;
         return 1;
      }
   }
   delete [] ttfont;

   fFontName[fFontCount] = StrDup(basename);
   fCurFontIdx = fFontCount;
   fFace[fCurFontIdx] = tface;
   fFontCount++;

   // get face properties and allocate preload arrays
   fProperties[fCurFontIdx] = new TT_Face_Properties;
   error = TT_Get_Face_Properties(*tface, fProperties[fCurFontIdx]);
   if (error) {
      SafeDelete(fProperties[fCurFontIdx]);
      Error("SetTextFont", "error getting properties for font %s", basename);
      goto fail;
   }

   // create glyph
   fGlyph[fCurFontIdx] = new TT_Glyph;
   error = TT_New_Glyph(*tface, fGlyph[fCurFontIdx]);
   if (error) {
      SafeDelete(fGlyph[fCurFontIdx]);
      Error("SetTextFont", "error creating glyph for font %s", basename);
      goto fail;
   }

   // create instance
   fInstance[fCurFontIdx] = new TT_Instance;
   error = TT_New_Instance(*tface, fInstance[fCurFontIdx]);
   if (error) {
      SafeDelete(fInstance[fCurFontIdx]);
      Error("SetTextFont", "error creating instance for font %s", basename);
      goto fail;
   }

   // set device resolution
   error = TT_Set_Instance_Resolutions(*fInstance[fCurFontIdx], res, res);
   if (error) {
      Error("SetTextFont", "could not set device resolutions for font %s", basename);
      goto fail;
   }

   fCharMap[fCurFontIdx] = 0;

   // set initial text size
   SetTextSize(fTextSize);

   return 0;

fail:
   TT_Close_Face(*tface);
   fFontCount--;
   delete [] fFontName[fFontCount];
   SafeDelete(fFace[fCurFontIdx]);
   if (mode == kCheck) return 1;
   if (fFontCount) {
      Warning("SetTextFont", "using default font %s", fFontName[0]);
      fCurFontIdx = 0;    // use font 0 (default font, set in ctor)
      return 0;
   } else {
      Error("SetTextFont", "switching back to X11 fonts");
      fHasTTFonts = kFALSE;
      return 1;
   }
}

//______________________________________________________________________________
void TGX11TTF::SetTextFont(Font_t fontnumber)
{
   // Set specified font.
   // List of the currently supported fonts (screen and PostScript)
   // =============================================================
   //   Font ID       X11                        TTF
   //        1 : times-medium-i-normal       timesi.ttf
   //        2 : times-bold-r-normal         timesbd.ttf
   //        3 : times-bold-i-normal         timesi.ttf
   //        4 : helvetica-medium-r-normal   arial.ttf
   //        5 : helvetica-medium-o-normal   ariali.ttf
   //        6 : helvetica-bold-r-normal     arialbd.ttf
   //        7 : helvetica-bold-o-normal     arialbi.ttf
   //        8 : courier-medium-r-normal     cour.ttf
   //        9 : courier-medium-o-normal     couri.ttf
   //       10 : courier-bold-r-normal       courbd.ttf
   //       11 : courier-bold-o-normal       courbi.ttf
   //       12 : symbol-medium-r-normal      symbol.ttf
   //       13 : times-medium-r-normal       times.ttf
   //       14 :                             wingding.ttf

#ifndef R__TTFROT
   if (!fHasTTFonts || fTextAngle != 0.) {
#else
   if (!fHasTTFonts) {
#endif
      TGX11::SetTextFont(fontnumber);
      return;
   }

   fTextFont = fontnumber;

   const char *fontname;

   switch (fontnumber/10) {

      case 1:
          fontname = "timesi.ttf";
          break;
      case 2:
          fontname = "timesbd.ttf";
          break;
      case 3:
          fontname = "timesbi.ttf";
          break;
      case 4:
          fontname = "arial.ttf";
          break;
      case 5:
          fontname = "ariali.ttf";
          break;
      case 6:
          fontname = "arialbd.ttf";
          break;
      case 7:
          fontname = "arialbi.ttf";
          break;
      case 8:
          fontname = "cour.ttf";
          break;
      case 9:
          fontname = "couri.ttf";
          break;
      case 10:
          fontname = "courbd.ttf";
          break;
      case 11:
          fontname = "courbi.ttf";
          break;
      case 12:
          fontname = "symbol.ttf";
          break;
      case 13:
          fontname = "times.ttf";
          break;
      case 14:
          fontname = "wingding.ttf";
          break;
      default:
          fontname = "arialbd.ttf";
          break;
   }

   SetTextFont((char *)fontname, kLoad);
}

//______________________________________________________________________________
void TGX11TTF::SetTextSize(Float_t textsize)
{
   // Set current text size.

#ifndef R__TTFROT
   if (!fHasTTFonts || fTextAngle != 0.) {
#else
   if (!fHasTTFonts) {
#endif
      TGX11::SetTextSize(textsize);
#ifndef R__TTFROT
      if (!fHasTTFonts)
#endif
      return;
   }

   if (gDebug > 0)
      printf("SetTextSize: %f\n", textsize);

   fTextSize = textsize;

   if (fTextSize < 0) return;

   if (fCurFontIdx < 0 || fFontCount <= fCurFontIdx) {
      Error("SetTextSize", "current font index out of bounds");
      return;
   }

   Int_t tsize = TMath::Nint(fTextSize * kScale) << 6;
   TT_Error error = TT_Set_Instance_CharSize(*fInstance[fCurFontIdx], tsize);
   if (error)
      Error("SetTextSize", "could not set new size in instance");
}

//______________________________________________________________________________
TTChar *TGX11TTF::GetChar(UInt_t code, UInt_t size, Float_t angle, Bool_t force)
{
   // Get a TTChar corresponding to the char code, size and current font.
   // If force is true (the default) and the cache is full the lru char
   // will be removed from the cache. If false it returns 0.

   // look for char in cache
   TTChar *c = LookupChar(code, size, angle, fFontName[fCurFontIdx]);

   if (!c) {
      while (fCacheCount >= kCacheSize) {

         if (!force) return 0;

         // get least recent used char
         TTChar *lru = (TTChar *)fLRU->LastLink()->GetObject();

         // remove from lru list and hashtable
         fLRU->Remove(lru->fLink);
         fCharCache->Remove(lru);

         delete lru;
         fCacheCount--;
      }

      // create new char
      if (angle == 0.)
         c = AllocChar(code, size, fFontName[fCurFontIdx]);
      else
         c = AllocRotatedChar(code, size, angle, fFontName[fCurFontIdx]);
   }

   return c;
}

//______________________________________________________________________________
TTChar *TGX11TTF::LookupChar(UInt_t code, UInt_t size, Float_t angle, const char *fontname)
{
   // Find TTChar corresponding to the char code, size and font in cache.
   // Returns 0 if not in cache.

   TTChar tmp;
   tmp.fCode     = code;
   tmp.fSize     = size;
   tmp.fAngle    = angle;
   tmp.fFontName = fontname;

   TTChar *c = (TTChar *) fCharCache->FindObject(&tmp);

   if (c) {
      // when found in cache bring to front of lru list
      fLRU->Remove(c->fLink);
      fLRU->AddFirst(c);
      c->fLink = fLRU->FirstLink();
      fCacheHits++;
   } else
      fCacheMisses++;

   return c;
}

//______________________________________________________________________________
TTChar *TGX11TTF::AllocChar(UInt_t code, UInt_t size, const char *fontname)
{
   // Allocate a TTChar object corresponding to the char code, size and font.
   // TTChar is added to the cache for later reuse. In case of failure
   // return 0.

   Short_t unicode = CharToUnicode(code);

   if (!unicode)
      return 0;

   TT_Error error;

   if ((error = LoadTrueTypeChar(unicode))) {
      Error("AllocChar", "LoadTrueTypeChar %c (0x%04x) failed for font %s (err=%d)",
             isprint(unicode) ? unicode : '?', unicode, fontname, error);
      //fprintf(stderr, "Reason: %s\n", TT_ErrToString18(error));
      return 0;
   }

   TT_Glyph_Metrics metrics;
   TT_Get_Glyph_Metrics(*fGlyph[fCurFontIdx], &metrics);

   // TT_Get_Glyph_Pixmap generates pixmap starting from FreeType
   // origin (leftmost side, baseline). Because of this 3rd and 4th
   // arguments are necessary.
   // For X axis (3rd argument), we have to take metrics.bearingX as
   // offset. Y axis must be shifted if there's descent box (image
   // below the baseline). 4th argument is specified for that.

   TT_Raster_Map bitmap;

   bitmap.rows = int((metrics.bbox.yMax - metrics.bbox.yMin) >> 6);
   bitmap.rows += 2;    // insurance to cope with number-round error
   bitmap.width = int((metrics.bbox.xMax - metrics.bbox.xMin) >> 6);
   bitmap.width += 2;   // insurance to cope with number-round error
   if (fSmoothing)
      bitmap.cols = (bitmap.width+3) & -4;
   else
      bitmap.cols = (bitmap.width+7) >> 3;
   bitmap.flow   = TT_Flow_Down;
   bitmap.size   = (long)bitmap.cols * bitmap.rows;
   bitmap.bitmap = (void *)new char[bitmap.size];
   memset(bitmap.bitmap, 0, (size_t)bitmap.size);

   // be very careful about TTCeil/TTFloor, must be multiples of 64 otherwise
   // hinting will be ruined if this is not the case

   if (fSmoothing)
      TT_Get_Glyph_Pixmap(*fGlyph[fCurFontIdx], &bitmap, TTCeil(-metrics.bearingX),
             TTCeil(metrics.bbox.yMax - metrics.bbox.yMin - metrics.bearingY));
   else
      TT_Get_Glyph_Bitmap(*fGlyph[fCurFontIdx], &bitmap, TTCeil(-metrics.bearingX),
             TTCeil(metrics.bbox.yMax - metrics.bbox.yMin - metrics.bearingY));

   TTChar *c = new TTChar;

   c->fCode   = code;
   c->fSize   = size;
   c->fAngle  = 0.;
   c->fWidth  = bitmap.width;
   c->fBwidth = bitmap.cols;
   c->fHeight = bitmap.rows;

   // ascent must be derived from descent, to avoid number-rounding errors
   c->fDescent  = int(TTCeil(metrics.bbox.yMax - metrics.bbox.yMin - metrics.bearingY) >> 6);
   c->fAscent   = bitmap.rows - c->fDescent;
   c->fXoff     = int(metrics.bearingX >> 6);
   if (c->fXoff < 0) c->fXoff = 0;
   c->fFontName = fontname;
   c->fCharlen  = int(metrics.advance >> 6);
   c->fBitmap   = (UChar_t *)bitmap.bitmap;

   fCharCache->Add(c);
   fLRU->AddFirst(c);
   c->fLink = fLRU->FirstLink();

   fCacheCount++;

   return c;
}

//______________________________________________________________________________
TTChar *TGX11TTF::AllocRotatedChar(UInt_t code, UInt_t size, Float_t angle,
                                   const char *fontname)
{
   // Allocate a TTChar object corresponding to the char code, size, angle
   // and font. TTChar is added to the cache for later reuse. In case of failure
   // return 0.

   Short_t unicode = CharToUnicode(code);

   if (!unicode)
      return 0;

   TT_Error error;

   if ((error = LoadTrueTypeChar(unicode))) {
      Error("AllocChar", "LoadTrueTypeChar %c (0x%04x) failed for font %s (err=%d)",
             isprint(unicode) ? unicode : '?', unicode, fontname, error);
      //fprintf(stderr, "Reason: %s\n", TT_ErrToString18(error));
      return 0;
   }

   TT_Glyph_Metrics metrics;
   TT_Get_Glyph_Metrics(*fGlyph[fCurFontIdx], &metrics);

   // access its outline
   TT_Outline outline;
   TT_Get_Glyph_Outline(*fGlyph[fCurFontIdx], &outline);

   // copy outline and rotate it
   TT_Outline toutline;
   TT_New_Outline(outline.n_points, outline.n_contours, &toutline);
   TT_Copy_Outline(&outline, &toutline);
   TT_Transform_Outline(&toutline, fRotMatrix);

   // get bounding box and grid-fit it
   TT_BBox bbox;
   TT_Get_Outline_BBox(&toutline, &bbox);

   bbox.xMin = TTFloor(bbox.xMin);
   bbox.yMin = TTFloor(bbox.yMin);
   bbox.xMax = TTCeil(bbox.xMax);
   bbox.yMax = TTCeil(bbox.yMax);

   TT_Translate_Outline(&toutline, -bbox.xMin, -bbox.yMin);

   // TT_Get_Outline_Pixmap generates pixmap starting from FreeType
   // origin (leftmost side, baseline). Because of this 3rd and 4th
   // arguments are necessary.
   // For X axis (3rd argument), we have to take metrics.bearingX as
   // offset. Y axis must be shifted if there's descent box (image
   // below the baseline). 4th argument is specified for that.

   TT_Raster_Map bitmap;

   bitmap.rows = int((bbox.yMax - bbox.yMin) >> 6);
   bitmap.rows += 2;    // insurance to cope with number-round error
   bitmap.width = int((bbox.xMax - bbox.xMin) >> 6);
   bitmap.width += 2;   // insurance to cope with number-round error
   if (fSmoothing)
      bitmap.cols = (bitmap.width+3) & -4;
   else
      bitmap.cols = (bitmap.width+7) >> 3;
   bitmap.flow   = TT_Flow_Down;
   bitmap.size   = (long)bitmap.cols * bitmap.rows;
   bitmap.bitmap = (void *)new char[bitmap.size];
   memset(bitmap.bitmap, 0, (size_t)bitmap.size);

   // be very careful about TTCeil/TTFloor, must be multiples of 64 otherwise
   // hinting will be ruined if this is not the case

   if (fSmoothing)
      TT_Get_Outline_Pixmap(*fEngine, &toutline, &bitmap);
   else
      TT_Get_Outline_Bitmap(*fEngine, &toutline, &bitmap);

   TTChar *c = new TTChar;

   c->fCode   = code;
   c->fSize   = size;
   c->fAngle  = angle;
   c->fWidth  = bitmap.width;
   c->fBwidth = bitmap.cols;
   c->fHeight = bitmap.rows;

   // ascent must be derived from descent, to avoid number-rounding errors
   c->fDescent  = int(bbox.yMin >> 6);
   c->fAscent   = bitmap.rows - c->fDescent;
   c->fXoff     = int(bbox.xMin >> 6);
   c->fFontName = fontname;
   c->fCharlen  = int(metrics.advance >> 6);
   c->fBitmap   = (UChar_t *)bitmap.bitmap;

   fCharCache->Add(c);
   fLRU->AddFirst(c);
   c->fLink = fLRU->FirstLink();

   fCacheCount++;

   TT_Done_Outline(&toutline);

   return c;
}

//______________________________________________________________________________
Short_t TGX11TTF::CharToUnicode(UInt_t code)
{
   // Map char to unicode. Returns 0 in case no mapping exists.

   if (!fCharMap[fCurFontIdx]) {
      UShort_t i, platform, encoding;

      fCharMap[fCurFontIdx] = new TT_CharMap;

      // first look for a unicode charmap
      Int_t n = fProperties[fCurFontIdx]->num_CharMaps;

      for (i = 0; i < n; i++) {
         TT_Get_CharMap_ID(*fFace[fCurFontIdx], i, &platform, &encoding);

         if ((platform == 3 && encoding == 1) ||
             (platform == 0 && encoding == 0) ||
             (platform == 1 && encoding == 0 &&
              !strcmp(fFontName[fCurFontIdx],"symbol.ttf"))) {
            TT_Get_CharMap(*fFace[fCurFontIdx], i, fCharMap[fCurFontIdx]);
            return TT_Char_Index(*fCharMap[fCurFontIdx], (Short_t) code);
         }
      }

      if (gDebug > 0)
         Warning("CharToUnicode", "the fontfile %s doesn't contain any unicode "
                 "mapping table\n", fFontName[fCurFontIdx]);
   }

   return TT_Char_Index(*fCharMap[fCurFontIdx], (Short_t) code);
}

//______________________________________________________________________________
Int_t TGX11TTF::LoadTrueTypeChar(Int_t idx)
{
   // Load the True Type character.

   int flags;

   flags = TTLOAD_SCALE_GLYPH;
   if (fHinting)
      flags |= TTLOAD_HINT_GLYPH;

   return TT_Load_Glyph(*fInstance[fCurFontIdx], *fGlyph[fCurFontIdx], idx, flags);
}

//______________________________________________________________________________
XImage *TGX11TTF::GetBackground(Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Get the background of the current window in an XImage.

   XWindow_t *cws = GetCurrentWindow();

   if (x < 0) {
      w += x;
      x = 0;
   }
   if (y < 0) {
      h += y;
      y = 0;
   }

   if (x+w > cws->width)  w = cws->width - x;
   if (y+h > cws->height) h = cws->height - y;

   return XGetImage(fDisplay, cws->drawing, x, y, w, h, AllPlanes, ZPixmap);
}

//______________________________________________________________________________
Bool_t TGX11TTF::IsVisible(Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // If string falls outside window, there is probably no need to draw it.

   XWindow_t *cws = GetCurrentWindow();

   if (x + (int)w <= 0 || x >= (int)cws->width)  return kFALSE;
   if (y + (int)h <= 0 || y >= (int)cws->height) return kFALSE;

   return kTRUE;
}

//______________________________________________________________________________
void TGX11TTF::DrawImage(TTChar *c, ULong_t fore, ULong_t back, XImage *xim,
                         Int_t bx, Int_t by)
{
   // Draw TTChar bitmap to xim image at position bx,by using specified
   // foreground color.

   UChar_t d = 0, *s = c->fBitmap;

   if (fSmoothing) {

      static XColor col[5];
      XColor  *bcol = 0, *bc;
      Int_t    x, y;
      //UInt_t   charlen = c->fCharlen;
      UInt_t   charlen = c->fWidth;

      // background kClear, i.e. transparent, we take as background color
      // the average of the rgb values of all pixels covered by this character
      if (back == (ULong_t) -1 && charlen) {
         ULong_t r, g, b;
         Int_t   dots, dotcnt;
         const Int_t maxdots = 50000;

         dots = Int_t(charlen * c->fHeight);
         dots = dots > maxdots ? maxdots : dots;
         bcol = new XColor[dots];
         if (!bcol) return;
         bc = bcol;
         dotcnt = 0;
         for (y = 0; y < (int) c->fHeight; y++) {
            for (x = 0; x < (int) charlen; x++, bc++) {
               bc->pixel = XGetPixel(xim, bx + c->fXoff + x, by - c->fAscent + y);
               bc->flags = DoRed|DoGreen|DoBlue;
               if (++dotcnt >= maxdots) break;
            }
         }
         XQueryColors(fDisplay, fColormap, bcol, dots);
         r = g = b = 0;
         bc = bcol;
         dotcnt = 0;
         for (y = 0; y < (int) c->fHeight; y++) {
            for (x = 0; x < (int) charlen; x++, bc++) {
               r += bc->red;
               g += bc->green;
               b += bc->blue;
               if (++dotcnt >= maxdots) break;
            }
         }
         r /= dots; g /= dots; b /= dots;
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
            col[3].flags = DoRed|DoGreen|DoBlue;
            XQueryColors(fDisplay, fColormap, &col[3], 2);
            col[0] = col[3];
         } else
            XQueryColor(fDisplay, fColormap, &col[4]);

         // interpolate between fore and backgound colors
         for (x = 3; x > 0; x--) {
            col[x].red   = (col[4].red  *x + col[0].red  *(4-x)) /4;
            col[x].green = (col[4].green*x + col[0].green*(4-x)) /4;
            col[x].blue  = (col[4].blue *x + col[0].blue *(4-x)) /4;
            if (!XAllocColor(fDisplay, fColormap, &col[x])) {
               Warning("DrawImage", "cannot allocate smoothing color");
               col[x].pixel = col[x+1].pixel;
            }
         }
      }

      // put smoothed character, character pixmap values are an index
      // into the 5 colors used for aliasing (4 = foreground, 0 = background)
      for (y = 0; y < (int) c->fHeight; y++) {
         for (x = 0; x < (int) c->fBwidth; x++) {
            d = *s++;
            if (d && x < (int) c->fWidth) {
               ULong_t p = col[d].pixel;
               XPutPixel(xim, bx + c->fXoff + x, by - c->fAscent + y, p);
            }
         }
      }

   } else {

      // no smoothing, just put charachter using foreground color
      for (int y = 0; y < (int) c->fHeight; y++) {
         int n = 0;
         for (int x = 0; x < (int) c->fWidth; x++) {
            if (n == 0) d = *s++;
            if (TESTBIT(d,7-n)) {
               XPutPixel(xim, bx + c->fXoff + x, by - c->fAscent + y, fore);
            }
            if (++n == (int) kBitsPerByte) n = 0;
         }
      }
   }
}

//______________________________________________________________________________
void TGX11TTF::SetRotationMatrix(Float_t angle)
{
   // Set the rotation matrix used to rotate the font outlines.

   //TT_Set_Instance_Transform_Flags( instance, 1, 0 );

   if (fRotMatrix) {
      SafeDelete(fRotMatrix);
   }
   fRotMatrix = new TT_Matrix;

   angle = Float_t(angle * TMath::Pi() / 180.);

   fRotMatrix->xx = (TT_Fixed) (TMath::Cos(angle) * (1<<16));
   fRotMatrix->xy = (TT_Fixed) (TMath::Sin(angle) * (1<<16));
   fRotMatrix->yx = -fRotMatrix->xy;
   fRotMatrix->yy =  fRotMatrix->xx;
}

//______________________________________________________________________________
void TGX11TTF::SetHinting(Bool_t state)
{
   // Set hinting flag. If status changes clear cache.

   if (fHinting != state) {
      ClearCache();
      fHinting = state;
   }
}

//______________________________________________________________________________
void TGX11TTF::SetSmoothing(Bool_t state)
{
   // Set smoothing (anit-aliasing) flag. If status changes clear cache.

   if (state && DefaultDepth(fDisplay,fScreenNumber) <= 8)
      Warning("SetSmoothing", "the colormap might not have enough free color "
                              "cells to fully support smoothing");

   if (fSmoothing != state) {
      ClearCache();
      fSmoothing = state;
   }
}
