// @(#)root/x11ttf:$Name:  $:$Id: TGWin32TTF.h,v 1.2 2001/02/17 11:42:23 rdm Exp $
// Author: Olivier Couet     01/10/02
// Author: Fons Rademakers   21/11/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGWin32TTF
#define ROOT_TGWin32TTF


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGWin32TTF                                                           //
//                                                                      //
// Interface to low level X11 (Xlib). This class gives access to basic  //
// X11 graphics via the parent class TGWin32. However, all text and     //
// font handling is done via the Freetype TrueType library. When the    //
// shared library containing this class is loaded the global gVirtualX  //
// is redirected to point to this class.                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGWIN32
#include "TGWin32.h"
#endif

#ifndef ROOT_TTF
#include "TTF.h"
#endif


class TGWin32TTF : public TGWin32 {

private:
   enum EAlign { kNone, kTLeft, kTCenter, kTRight, kMLeft, kMCenter, kMRight,
                        kBLeft, kBCenter, kBRight };

   FT_Vector   fAlign;                 // alignment vector

   void    Align(void);
   void    DrawImage(FT_Bitmap *source, ULong_t fore, ULong_t back, GdkImage *xim,
                     Int_t bx, Int_t by);
   Bool_t  IsVisible(Int_t x, Int_t y, UInt_t w, UInt_t h);
   GdkImage *GetBackground(Int_t x, Int_t y, UInt_t w, UInt_t h);
   void    RenderString(Int_t x, Int_t y, ETextMode mode);

public:
   TGWin32TTF(const TGWin32 &org);
   virtual ~TGWin32TTF();

   void   DrawText(Int_t x, Int_t y, Float_t angle, Float_t mgn,
                   const char *text, ETextMode mode);
   void   SetTextFont(Font_t fontnumber);
   Int_t  SetTextFont(char *fontname, ETextSetMode mode);
   void   SetTextSize(Float_t textsize);

   ClassDef(TGWin32TTF,0)  //Interface to X11 + TTF font handling
};

#endif
