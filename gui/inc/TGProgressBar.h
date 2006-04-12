// @(#)root/gui:$Name:  $:$Id: TGProgressBar.h,v 1.9 2004/09/08 08:13:11 brun Exp $
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

public:
   enum EBarType { kStandard, kFancy };
   enum EFillType { kSolidFill, kBlockFill };
   enum { kProgressBarStandardWidth = 16, kProgressBarTextWidth = 24,
          kBlockSize = 8, kBlockSpace = 2 };

protected:
   Float_t       fMin;          // logical minimum value (default 0)
   Float_t       fMax;          // logical maximum value (default 100)
   Float_t       fPos;          // logical position [fMin,fMax]
   Int_t         fPosPix;       // position of progress bar in pixel coordinates
   Int_t         fBarWidth;     // progress bar width
   EFillType     fType;         // fill type (default kSolidFill)
   EBarType      fBarType;      // bar type (default kStandard)
   TString       fFormat;       // format used to show position not in percent
   Bool_t        fShowPos;      // show position value (default false)
   Bool_t        fPercent;      // show position in percent (default true)
   Bool_t        fDrawBar;      // if true draw only bar in DoRedraw()
   TGGC          fBarColorGC;   // progress bar drawing context
   GContext_t    fNormGC;       // text drawing graphics context
   FontStruct_t  fFontStruct;   // font used to draw position text

   virtual void DoRedraw() = 0;

   static const TGFont *fgDefaultFont;
   static TGGC         *fgDefaultGC;

public:
   static FontStruct_t  GetDefaultFontStruct();
   static const TGGC   &GetDefaultGC();

   TGProgressBar(const TGWindow *p, UInt_t w, UInt_t h,
                 Pixel_t back = GetWhitePixel(),
                 Pixel_t barcolor = GetDefaultSelectedBackground(),
                 GContext_t norm = GetDefaultGC()(),
                 FontStruct_t font = GetDefaultFontStruct(),
                 UInt_t options = kDoubleBorder | kSunkenFrame);
   virtual ~TGProgressBar() { }

   void         SetRange(Float_t min, Float_t max);   //*MENU*
   void         SetPosition(Float_t pos);             //*MENU*
   void         Increment(Float_t inc);
   void         Reset();                              //*MENU*
   void         SetFillType(EFillType type);          //*MENU*
   void         SetBarColor(Pixel_t color);
   void         SetBarColor(const char *color);       //*MENU*
   Float_t      GetMin() const { return fMin; }
   Float_t      GetMax() const { return fMax; }
   Float_t      GetPosition() const { return fPos; }
   EFillType    GetFillType() const { return fType; }
   EBarType     GetBarType() const { return fBarType; }
   Bool_t       GetShowPos() const { return fShowPos; }
   TString      GetFormat() const { return fFormat; }
   Bool_t       UsePercent() const { return fPercent; }
   virtual void SavePrimitive(ofstream &out, Option_t *option);

   ClassDef(TGProgressBar,0)  // Progress bar abstract base class
};


class TGHProgressBar : public TGProgressBar {

protected:
   virtual void DoRedraw();

public:
   TGHProgressBar(const TGWindow *p = 0,
                  UInt_t w = 4, UInt_t h = kProgressBarTextWidth,
                  Pixel_t back = GetWhitePixel(),
                  Pixel_t barcolor = GetDefaultSelectedBackground(),
                  GContext_t norm = GetDefaultGC()(),
                  FontStruct_t font = GetDefaultFontStruct(),
                  UInt_t options = kDoubleBorder | kSunkenFrame) :
      TGProgressBar(p, w, h, back, barcolor, norm, font, options) { fBarWidth = h; }
   TGHProgressBar(const TGWindow *p, EBarType type, UInt_t w);
   virtual ~TGHProgressBar() { }

   virtual TGDimension GetDefaultSize() const
                     { return TGDimension(fWidth, fBarWidth); }

   void ShowPosition(Bool_t set = kTRUE, Bool_t percent = kTRUE,
                     const char *format = "%.2f");          //*MENU*
   virtual void SavePrimitive(ofstream &out, Option_t *option);

   ClassDef(TGHProgressBar,0)  // Horizontal progress bar widget
};


class TGVProgressBar : public TGProgressBar {

protected:
   virtual void DoRedraw();

public:
   TGVProgressBar(const TGWindow *p = 0,
                  UInt_t w = kProgressBarTextWidth, UInt_t h = 4,
                  Pixel_t back = GetWhitePixel(),
                  Pixel_t barcolor = GetDefaultSelectedBackground(),
                  GContext_t norm = GetDefaultGC()(),
                  FontStruct_t font = GetDefaultFontStruct(),
                  UInt_t options = kDoubleBorder | kSunkenFrame) :
      TGProgressBar(p, w, h, back, barcolor, norm, font, options) { fBarWidth = w; }
   TGVProgressBar(const TGWindow *p, EBarType type, UInt_t h);
   virtual ~TGVProgressBar() { }

   virtual TGDimension GetDefaultSize() const
                     { return TGDimension(fBarWidth, fHeight); }
   virtual void SavePrimitive(ofstream &out, Option_t *option);

   ClassDef(TGVProgressBar,0)  // Vertical progress bar widget
};

#endif

