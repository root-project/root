// @(#)root/gui:$Name:  $:$Id: TGProgressBar.h,v 1.1 2000/10/09 19:13:29 rdm Exp $
// Author: Fons Rademakers   10/10/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGProgressBar
#define ROOT_TGProgressBar


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGProgressBar, TGHProgressBar and TGVProgressBar                     //
//                                                                      //
// The classes in this file implement progress bars. Progress bars can  //
// be used to show progress of tasks taking more then a few seconds.    //
// TGProgressBar is an abstract base class, use either TGHProgressBar   //
// or TGVProgressBar. TGHProgressBar can in addition show the position  //
// as text in the bar.                                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif


class TGProgressBar : public TGFrame {

friend class TGClient;

public:
   enum EFillType { kSolidFill, kBlockFill };
   enum { kProgressBarWidth = 24 };

protected:
   Float_t       fMin;          // logical minimum value (default 0)
   Float_t       fMax;          // logical maximum value (default 100)
   Float_t       fPos;          // logical position [fMin,fMax]
   Int_t         fPosPix;       // position of progress bar in pixel coordinates
   Int_t         fBarWidth;     // progress bar width
   EFillType     fType;         // fill type (default kSolidFill)
   TString       fFormat;       // format used to show position not in percent
   Bool_t        fShowPos;      // show position value (default false)
   Bool_t        fPercent;      // show position in percent (default true)
   Bool_t        fDrawBar;      // if true draw only bar in DoRedraw()
   TGGC          fBarColorGC;   // progress bar drawing context
   GContext_t    fNormGC;       // text drawing graphics context
   FontStruct_t  fFontStruct;   // font used to draw position text

   virtual void DoRedraw() = 0;

   static FontStruct_t  fgDefaultFontStruct;
#ifdef R__SUNCCBUG
public:
#endif
   static TGGC          fgDefaultGC;
   static TGGC          fgDefaultBarColorGC;

public:
   TGProgressBar(const TGWindow *p, UInt_t w, UInt_t h,
                 ULong_t back = fgWhitePixel,
                 ULong_t barcolor = fgDefaultSelectedBackground,
                 GContext_t norm = fgDefaultGC(),
                 FontStruct_t font = fgDefaultFontStruct,
                 UInt_t options = kDoubleBorder | kSunkenFrame);
   virtual ~TGProgressBar() { }

   void         SetRange(Float_t min, Float_t max);
   void         SetPosition(Float_t pos);
   void         Increment(Float_t inc);
   void         Reset();
   void         SetFillType(EFillType type);
   void         SetBarColor(ULong_t color);
   void         SetBarColor(const char *color);
   Float_t      GetMin() const { return fMin; }
   Float_t      GetMax() const { return fMax; }
   Float_t      GetPosition() const { return fPos; }
   EFillType    GetFillType() const { return fType; }
   Bool_t       UsePercent() const { return fPercent; }

   static FontStruct_t  GetDefaultFontStruct();
   static const TGGC   &GetDefaultGC();

   ClassDef(TGProgressBar,0)  // Progress bar abstract base class
};


class TGHProgressBar : public TGProgressBar {

protected:
   virtual void DoRedraw();

public:
   TGHProgressBar(const TGWindow *p, UInt_t w = 4, UInt_t h = kProgressBarWidth,
                  ULong_t back = fgWhitePixel,
                  ULong_t barcolor = fgDefaultSelectedBackground,
                  GContext_t norm = fgDefaultGC(),
                  FontStruct_t font = fgDefaultFontStruct,
                  UInt_t options = kDoubleBorder | kSunkenFrame) :
      TGProgressBar(p, w, h, back, barcolor, norm, font, options) { fBarWidth = h; }
   virtual ~TGHProgressBar() { }

   virtual TGDimension GetDefaultSize() const
                     { return TGDimension(fWidth, fBarWidth); }

   void ShowPosition(Bool_t set = kTRUE, Bool_t percent = kTRUE,
                     const char *format = "%.2f");

   ClassDef(TGHProgressBar,0)  // Horizontal progress bar widget
};


class TGVProgressBar : public TGProgressBar {

protected:
   virtual void DoRedraw();

public:
   TGVProgressBar(const TGWindow *p, UInt_t w = kProgressBarWidth, UInt_t h = 4,
                  ULong_t back = fgWhitePixel,
                  ULong_t barcolor = fgDefaultSelectedBackground,
                  GContext_t norm = fgDefaultGC(),
                  FontStruct_t font = fgDefaultFontStruct,
                  UInt_t options = kDoubleBorder | kSunkenFrame) :
      TGProgressBar(p, w, h, back, barcolor, norm, font, options) { fBarWidth = w; }
   virtual ~TGVProgressBar() { }

   virtual TGDimension GetDefaultSize() const
                     { return TGDimension(fBarWidth, fHeight); }

   ClassDef(TGVProgressBar,0)  // Vertical progress bar widget
};

#endif

