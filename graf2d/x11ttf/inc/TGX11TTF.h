// @(#)root/x11ttf:$Id$
// Author: Olivier Couet     01/10/02
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
// shared library containing this class is loaded the global gVirtualX  //
// is redirected to point to this class.                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGX11
#include "TGX11.h"
#endif

#ifndef ROOT_TTF
#include "TTF.h"
#endif

#ifndef ROOT_RConfigure
#include "RConfigure.h"
#endif

#ifdef R__HAS_XFT
class TXftFontHash;
#endif

class TGX11TTF : public TGX11 {

private:
   enum EAlign { kNone, kTLeft, kTCenter, kTRight, kMLeft, kMCenter, kMRight,
                        kBLeft, kBCenter, kBRight };

   FT_Vector   fAlign;                 // alignment vector
#ifdef R__HAS_XFT
   TXftFontHash  *fXftFontHash;        // hash table for Xft fonts
#endif

   void     Align(void);
   void     DrawImage(FT_Bitmap *source, ULong_t fore, ULong_t back, RXImage *xim,
                      Int_t bx, Int_t by);
   Bool_t   IsVisible(Int_t x, Int_t y, UInt_t w, UInt_t h);
   RXImage *GetBackground(Int_t x, Int_t y, UInt_t w, UInt_t h);
   void     RenderString(Int_t x, Int_t y, ETextMode mode);

public:
   TGX11TTF(const TGX11 &org);
   virtual ~TGX11TTF() { }

   Bool_t Init(void *display);
   void   DrawText(Int_t x, Int_t y, Float_t angle, Float_t mgn,
                   const char *text, ETextMode mode);
   void   DrawText(Int_t x, Int_t y, Float_t angle, Float_t mgn,
                   const wchar_t *text, ETextMode mode);
   void   SetTextFont(Font_t fontnumber);
   Int_t  SetTextFont(char *fontname, ETextSetMode mode);
   void   SetTextSize(Float_t textsize);

#ifdef R__HAS_XFT
   //---- Methods used text/fonts handling via Xft -----
   //void         SetClipRectangles(GContext_t gc, Int_t x, Int_t y, Rectangle_t *recs, Int_t n);
   FontStruct_t LoadQueryFont(const char *font_name);
   void         DeleteFont(FontStruct_t fs);
   void         DeleteGC(GContext_t gc);
   void         DrawString(Drawable_t id, GContext_t gc, Int_t x, Int_t y, const char *s, Int_t len);
   Int_t        TextWidth(FontStruct_t font, const char *s, Int_t len);
   void         GetFontProperties(FontStruct_t font, Int_t &max_ascent, Int_t &max_descent);
   FontH_t      GetFontHandle(FontStruct_t fs);
   FontStruct_t GetGCFont(GContext_t gc);
   void         MapGCFont(GContext_t gc, FontStruct_t font);
#endif

   static void  Activate();

   ClassDef(TGX11TTF,0)  //Interface to X11 + TTF font handling
};

#endif
