// @(#)root/gui:$Id$
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


#include "TGFrame.h"


class TGProgressBar : public TGFrame {

public:
   enum EBarType { kStandard, kFancy };
   enum EFillType { kSolidFill, kBlockFill };
   enum { kProgressBarStandardWidth = 16, kProgressBarTextWidth = 24,
          kBlockSize = 8, kBlockSpace = 2 };

protected:
   Float_t       fMin;          ///< logical minimum value (default 0)
   Float_t       fMax;          ///< logical maximum value (default 100)
   Float_t       fPos;          ///< logical position [fMin,fMax]
   Int_t         fPosPix;       ///< position of progress bar in pixel coordinates
   Int_t         fBarWidth;     ///< progress bar width
   EFillType     fFillType;     ///< *OPTION={GetMethod="GetFillType";SetMethod="SetFillType";Items=(kSolidFill=Solid",kBlockFill="Block")}*
   EBarType      fBarType;      ///< *OPTION={GetMethod="GetBarType";SetMethod="SetBarType";Items=(kStandard="Standard",kFancy="Fancy")}*
   TString       fFormat;       ///< format used to show position not in percent
   Bool_t        fShowPos;      ///< show position value (default false)
   Bool_t        fPercent;      ///< show position in percent (default true)
   Bool_t        fDrawBar;      ///< if true draw only bar in DoRedraw()
   TGGC          fBarColorGC;   ///< progress bar drawing context
   GContext_t    fNormGC;       ///< text drawing graphics context
   FontStruct_t  fFontStruct;   ///< font used to draw position text

   virtual void DoRedraw() override = 0;

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

   Float_t      GetMin() const { return fMin; }
   Float_t      GetMax() const { return fMax; }
   Float_t      GetPosition() const { return fPos; }
   EFillType    GetFillType() const { return fFillType; }
   EBarType     GetBarType() const { return fBarType; }
   Bool_t       GetShowPos() const { return fShowPos; }
   TString      GetFormat() const { return fFormat; }
   const char*  GetValueFormat() const { return fFormat.Data(); }
   Bool_t       UsePercent() const { return fPercent; }
   Pixel_t      GetBarColor() const { return fBarColorGC.GetForeground(); }
   GContext_t   GetNormGC() const { return fNormGC; }
   FontStruct_t GetFontStruct() const { return fFontStruct; }

   void         SetPosition(Float_t pos);                //*MENU*  *GETTER=GetPosition
   void         SetRange(Float_t min, Float_t max);      //*MENU*
   void         Increment(Float_t inc);
   void         SetBarType(EBarType type);               //*SUBMENU*
   void         SetFillType(EFillType type);             //*SUBMENU*
   virtual void Percent(Bool_t on) { fPercent = on; fClient->NeedRedraw(this); } //*TOGGLE* *GETTER=UsePercent
   virtual void ShowPos(Bool_t on) { fShowPos = on; fClient->NeedRedraw(this); } //*TOGGLE* *GETTER=GetShowPos
   virtual void Format(const char *format = "%.2f");     //*MENU* *GETTER=GetValueFormat
   void         SetMin(Float_t min) { fMin = min; }
   void         SetMax(Float_t max) { fMax = max; }
   virtual void SetBarColor(Pixel_t color);
   void         SetBarColor(const char *color="blue");
   virtual void Reset();                                 //*MENU*
   void         SetForegroundColor(Pixel_t pixel) override;

   void         SavePrimitive(std::ostream &out, Option_t *option = "") override;

   ClassDefOverride(TGProgressBar,0)  // Progress bar abstract base class
};


class TGHProgressBar : public TGProgressBar {

protected:
   void DoRedraw() override;

public:
   TGHProgressBar(const TGWindow *p = nullptr,
                  UInt_t w = 4, UInt_t h = kProgressBarTextWidth,
                  Pixel_t back = GetWhitePixel(),
                  Pixel_t barcolor = GetDefaultSelectedBackground(),
                  GContext_t norm = GetDefaultGC()(),
                  FontStruct_t font = GetDefaultFontStruct(),
                  UInt_t options = kDoubleBorder | kSunkenFrame);
   TGHProgressBar(const TGWindow *p, EBarType type, UInt_t w);
   virtual ~TGHProgressBar() { }

   TGDimension GetDefaultSize() const override
               { return TGDimension(fWidth, fBarWidth); }

   void ShowPosition(Bool_t set = kTRUE, Bool_t percent = kTRUE,
                     const char *format = "%.2f");

   void SavePrimitive(std::ostream &out, Option_t *option = "") override;

   ClassDefOverride(TGHProgressBar,0)  // Horizontal progress bar widget
};


class TGVProgressBar : public TGProgressBar {

protected:
   void DoRedraw() override;

public:
   TGVProgressBar(const TGWindow *p = nullptr,
                  UInt_t w = kProgressBarTextWidth, UInt_t h = 4,
                  Pixel_t back = GetWhitePixel(),
                  Pixel_t barcolor = GetDefaultSelectedBackground(),
                  GContext_t norm = GetDefaultGC()(),
                  FontStruct_t font = GetDefaultFontStruct(),
                  UInt_t options = kDoubleBorder | kSunkenFrame);
   TGVProgressBar(const TGWindow *p, EBarType type, UInt_t h);
   virtual ~TGVProgressBar() { }

   TGDimension GetDefaultSize() const override
                { return TGDimension(fBarWidth, fHeight); }
   void SavePrimitive(std::ostream &out, Option_t *option = "") override;
   void ShowPos(Bool_t) override { }
   void Percent(Bool_t) override { }

   ClassDefOverride(TGVProgressBar,0)  // Vertical progress bar widget
};

#endif

