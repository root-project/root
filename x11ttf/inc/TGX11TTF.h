// @(#)root/x11ttf:$Name:  $:$Id: TGX11TTF.h,v 1.3 2003/01/20 08:44:47 brun Exp $
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


class TGX11TTF : public TGX11 {

private:
   enum EAlign { kNone, kTLeft, kTCenter, kTRight, kMLeft, kMCenter, kMRight,
                        kBLeft, kBCenter, kBRight };

   FT_Vector   fAlign;                 // alignment vector

   void    Align(void);
   void    DrawImage(FT_Bitmap *source, ULong_t fore, ULong_t back, XImage *xim,
                     Int_t bx, Int_t by);
   Bool_t  IsVisible(Int_t x, Int_t y, UInt_t w, UInt_t h);
   XImage *GetBackground(Int_t x, Int_t y, UInt_t w, UInt_t h);
   void    RenderString(Int_t x, Int_t y, ETextMode mode);

public:
   TGX11TTF(const TGX11 &org);
   virtual ~TGX11TTF();

   void   DrawText(Int_t x, Int_t y, Float_t angle, Float_t mgn,
                   const char *text, ETextMode mode);
   void   SetTextFont(Font_t fontnumber);
   Int_t  SetTextFont(char *fontname, ETextSetMode mode);
   void   SetTextSize(Float_t textsize);

   ClassDef(TGX11TTF,0)  //Interface to X11 + TTF font handling
};

#endif
