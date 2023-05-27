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


#include "TGX11.h"

#include "TTF.h"

#include "RConfigure.h"

#ifdef R__HAS_XFT
class TXftFontHash;
#endif

class TGX11TTF : public TGX11 {

private:
   enum EAlign { kNone, kTLeft, kTCenter, kTRight, kMLeft, kMCenter, kMRight,
                        kBLeft, kBCenter, kBRight };

   FT_Vector   fAlign;                 ///< alignment vector
#ifdef R__HAS_XFT
   TXftFontHash  *fXftFontHash;        ///< hash table for Xft fonts
#endif

   void     Align(void);
   void     DrawImage(FT_Bitmap *source, ULong_t fore, ULong_t back, RXImage *xim,
                      Int_t bx, Int_t by);
   Bool_t   IsVisible(Int_t x, Int_t y, UInt_t w, UInt_t h);
   RXImage *GetBackground(Int_t x, Int_t y, UInt_t w, UInt_t h);
   void     RenderString(Int_t x, Int_t y, ETextMode mode);

public:
   TGX11TTF(const TGX11 &org);
   ~TGX11TTF() override { }

   Bool_t Init(void *display) override;
   void   DrawText(Int_t x, Int_t y, Float_t angle, Float_t mgn,
                   const char *text, ETextMode mode) override;
   void   DrawText(Int_t x, Int_t y, Float_t angle, Float_t mgn,
                   const wchar_t *text, ETextMode mode) override;
   void   SetTextFont(Font_t fontnumber) override;
   Int_t  SetTextFont(char *fontname, ETextSetMode mode) override;
   void   SetTextSize(Float_t textsize) override;

#ifdef R__HAS_XFT
   //---- Methods used text/fonts handling via Xft -----
   //void         SetClipRectangles(GContext_t gc, Int_t x, Int_t y, Rectangle_t *recs, Int_t n);
   FontStruct_t LoadQueryFont(const char *font_name) override;
   void         DeleteFont(FontStruct_t fs) override;
   void         DeleteGC(GContext_t gc) override;
   void         DrawString(Drawable_t id, GContext_t gc, Int_t x, Int_t y, const char *s, Int_t len) override;
   Int_t        TextWidth(FontStruct_t font, const char *s, Int_t len) override;
   void         GetFontProperties(FontStruct_t font, Int_t &max_ascent, Int_t &max_descent) override;
   FontH_t      GetFontHandle(FontStruct_t fs) override;
   FontStruct_t GetGCFont(GContext_t gc) override;
   void         MapGCFont(GContext_t gc, FontStruct_t font) override;
#endif

   static void  Activate();

   ClassDefOverride(TGX11TTF,0)  //Interface to X11 + TTF font handling
};

#endif
