// @(#)root/base:$Name$:$Id$
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TColor
#define ROOT_TColor


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TColor                                                               //
//                                                                      //
// Color defined by RGB or HLS.                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class TColor : public TNamed {

private:
   Int_t        fNumber;        //color number identifier
   Float_t      fRed;           //Fraction of Red
   Float_t      fGreen;         //Fraction of Green
   Float_t      fBlue;          //Fraction of Blue
   Float_t      fHue;           //Hue
   Float_t      fLight;         //Light
   Float_t      fSaturation;    //Saturation

public:
   TColor();
   TColor(Int_t color, Float_t r, Float_t g, Float_t b, const char *name="");
   TColor(const TColor &color);
   virtual ~TColor();
           void  Copy(TObject &color);
   virtual void  GetRGB(Float_t &r, Float_t &g, Float_t &b) {r = fRed; g = fGreen; b = fBlue;}
   virtual void  GetHLS(Float_t &h, Float_t &l, Float_t &s) {h = fHue; l = fLight; s = fSaturation;}
   Int_t         GetNumber() {return fNumber;}
   Float_t       GetRed() {return fRed;}
   Float_t       GetGreen() {return fGreen;}
   Float_t       GetBlue() {return fBlue;}
   Float_t       GetHue() {return fHue;}
   Float_t       GetLight() {return fLight;}
   Float_t       GetSaturation() {return fSaturation;}
   virtual void  HLStoRGB(Float_t h, Float_t l, Float_t s, Float_t &r, Float_t &g, Float_t &b);
   Float_t       HLStoRGB1(Float_t rn1, Float_t rn2, Float_t huei);
   virtual void  ls(Option_t *option="");
   virtual void  Print(Option_t *option="");
   virtual void  RGBtoHLS(Float_t r, Float_t g, Float_t b, Float_t &h, Float_t &l, Float_t &s);
   virtual void  SetNumber(Int_t number) {fNumber = number;}
   virtual void  SetRGB(Float_t r, Float_t g, Float_t b);

   ClassDef(TColor,1)  //Color defined by RGB or HLS
};

#endif

