// @(#)root/x11ttf:$Name$:$Id$
// Author: Fons Rademakers   21/11/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGX11TTF
#define ROOT_TGX11TTF


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

#ifndef ROOT_TGX11
#include "TGX11.h"
#endif

#if !defined(__CINT__)

#if defined(R__HPUX) && !defined(R__ACC)
#define signed
#endif

#include <freetype.h>

#if defined(R__HPUX) && !defined(R__ACC)
#undef signed
#endif

#else

struct TT_Face_Properties;
struct TT_Face;
struct TT_Glyph;
struct TT_Instance;
struct TT_CharMap;
struct TT_Engine;
struct TT_Matrix;

#endif

class TList;
class THashTable;
class TTChar;


class TGX11TTF : public TGX11 {

private:
   enum { kTTMaxFonts = 32, kCacheSize = 3000, kHashSize = 255 };
   enum EAlign { kNone, kTLeft, kTCenter, kTRight, kMLeft, kMCenter, kMRight,
                        kBLeft, kBCenter, kBRight };

   Int_t               fFontCount;                // number of fonts loaded
   Int_t               fCurFontIdx;               // current font index
   char               *fFontName[kTTMaxFonts];    // font name
   TT_Face_Properties *fProperties[kTTMaxFonts];  // font properties
   TT_Face            *fFace[kTTMaxFonts];        // font face
   TT_Glyph           *fGlyph[kTTMaxFonts];       // font glyph
   TT_Instance        *fInstance[kTTMaxFonts];    // font instance
   TT_CharMap         *fCharMap[kTTMaxFonts];     // font character map
   TT_Engine          *fEngine;                   // TrueType font renderer
   TT_Matrix          *fRotMatrix;                // rotation matrix
   Int_t               fCacheCount;               // number of chars in cache
   Int_t               fCacheHits;                // number of cache hits
   Int_t               fCacheMisses;              // number of cache misses
   THashTable         *fCharCache;                // character cache
   TList              *fLRU;                      // Least Recent Used chars
   Bool_t              fHinting;                  // use hinting (true by default)
   Bool_t              fSmoothing;                // use anti-aliasing (true when >8 planes, false otherwise)

   TTChar *GetChar(UInt_t code, UInt_t size, Float_t angle, Bool_t force = kTRUE);
   TTChar *LookupChar(UInt_t code, UInt_t size, Float_t angle, const char *fontname);
   TTChar *AllocChar(UInt_t code, UInt_t size, const char *fontname);
   TTChar *AllocRotatedChar(UInt_t code, UInt_t size, Float_t angle, const char *fontname);
   Short_t CharToUnicode(UInt_t code);
   Int_t   LoadTrueTypeChar(Int_t idx);
   XImage *GetBackground(Int_t x, Int_t y, UInt_t w, UInt_t h);
   void    DrawRotatedText(Int_t x, Int_t y, Float_t angle, const char *text, ETextMode mode);
   void    DrawImage(TTChar *c, ULong_t fore, ULong_t back, XImage *xim, Int_t bx, Int_t by);
   void    GetTextExtent(UInt_t &w, UInt_t &h, Int_t &maxAscent, const char *text);
   void    GetRotatedTextExtent(UInt_t &w, UInt_t &h, Int_t &xoff, Int_t &yoff, Float_t angle, const char *text);
   void    Align(UInt_t w, UInt_t h, Int_t maxAscent, Int_t &x, Int_t &y);
   void    AlignRotated(UInt_t w, UInt_t h, Int_t xoff, Int_t yoff, Int_t &x, Int_t &y);
   Bool_t  IsVisible(Int_t x, Int_t y, UInt_t w, UInt_t h);
   void    ClearCache();
   void    SetRotationMatrix(Float_t angle);

public:
   TGX11TTF() { fCharCache = 0; fLRU = 0; }
   TGX11TTF(const TGX11 &org);
   virtual ~TGX11TTF();

   void  DrawText(Int_t x, Int_t y, Float_t angle, Float_t mgn, const char *text, ETextMode mode);
   void  GetTextExtent(UInt_t &w, UInt_t &h, char *text);
   Int_t SetTextFont(char *fontname, ETextSetMode mode);
   void  SetTextFont(Font_t fontnumber);
   void  SetTextSize(Float_t textsize);

   Int_t GetCacheCount() const { return fCacheCount; }
   Int_t GetCacheHits() const { return fCacheHits; }
   Int_t GetCacheMisses() const { return fCacheMisses; }

   Bool_t GetHinting() const { return fHinting; }
   Bool_t GetSmoothing() const { return fSmoothing; }
   void   SetHinting(Bool_t state);
   void   SetSmoothing(Bool_t state);

   ClassDef(TGX11TTF,0)  //Interface to X11 + TTF font handling
};

#endif
