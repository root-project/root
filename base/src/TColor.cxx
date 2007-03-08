// @(#)root/base:$Name:  $:$Id: TColor.cxx,v 1.30 2007/03/08 12:21:43 brun Exp $
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Riostream.h"
#include "TROOT.h"
#include "TColor.h"
#include "TObjArray.h"
#include "TVirtualPad.h"
#include "TVirtualX.h"
#include "TError.h"
#include "TMathBase.h"
#include "TApplication.h"
#include <cmath>

ClassImp(TColor)

Bool_t  TColor::fgGrayscaleMode = kFALSE;
Bool_t  TColor::fgInitDone = kFALSE;
TArrayI TColor::fgPalette(0);

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TColor                                                               //
//                                                                      //
// Color defined by RGB or HLS.                                         //
// At initialization time, a table of colors is generated. This linked  //
// list can be accessed from the ROOT object                            //
// (see TROOT::GetListOfColors()). When a color is defined in the range //
// of [1,50], two "companion" colors are also defined:                  //
//    - the dark version (color_index + 100)                            //
//    - the bright version (color_index + 150)                          //
// The dark and bright color are used to give 3-D effects when drawing  //
// various boxes (see TWbox, TPave, TPaveText, TPaveLabel,etc).         //
//                                                                      //
// This is the list of currently supported basic colors (here dark and  //
// bright colors are not shown).                                        //
//Begin_Html
/*
<img src="gif/colors.gif">
*/
//End_Html
//
// One can toggle between a grayscale preview and the regular           //
// colored mode using SetGrayscale(). Note that in grayscale mode,      //
// access via RGB will return grayscale values according to ITU         //
// standards (and close to b&w printer grayscales), while access via    //
// HLS returns de-saturated grayscales.                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
TColor::TColor(): TNamed()
{
   // Default ctor.

   fNumber = -1;
   fRed = fGreen = fBlue = fHue = fLight = fSaturation = -1;
   fAlpha = 1;
}

//______________________________________________________________________________
TColor::TColor(Int_t color, Float_t r, Float_t g, Float_t b, const char *name, 
               Float_t a)
      : TNamed(name,"")
{
   // Normal color constructor. Initialize a color structure.
   // Compute the RGB and HLS parameters

   TColor::InitializeColors();
   // do not enter if color number already exist
   TColor *col = gROOT->GetColor(color);
   if (col) {
      Warning("TColor", "color %d already defined", color);
      fNumber = col->GetNumber();
      fRed    = col->GetRed();
      fGreen  = col->GetGreen();
      fBlue   = col->GetBlue();
      fHue    = col->GetHue();
      fLight  = col->GetLight();
      fAlpha  = col->GetAlpha();
      fSaturation = col->GetSaturation();
      return;
   }

   fNumber = color;

   char aname[32];
   if (!name || !*name) {
      sprintf(aname, "Color%d", color);
      SetName(aname);
   }

   // enter in the list of colors
   TObjArray *lcolors = (TObjArray*)gROOT->GetListOfColors();
   lcolors->AddAtAndExpand(this, color);

   // fill color structure
   SetRGB(r, g, b);
   fAlpha = a;
}

//______________________________________________________________________________
TColor::~TColor()
{
   // Color destructor.

   gROOT->GetListOfColors()->Remove(this);
}

//______________________________________________________________________________
TColor::TColor(const TColor &color) : TNamed(color)
{
   // Color copy ctor.

   ((TColor&)color).Copy(*this);
}

//______________________________________________________________________________
void TColor::InitializeColors()
{
   // Initialize colors used by the TCanvas based graphics (via TColor objects).
   // This method should be called before the ApplicationImp is created (which
   // initializes the GUI colors).

   if (fgInitDone)
      return;
   fgInitDone = kTRUE;
   if (gROOT->GetListOfColors()->First() == 0) {
      TColor *s0;
      Float_t r, g, b, h, l, s;
      Int_t   i;

      new TColor(kWhite,1,1,1,"background");
      new TColor(kBlack,0,0,0,"black");
      new TColor(2,1,0,0,"red");
      new TColor(3,0,1,0,"green");
      new TColor(4,0,0,1,"blue");
      new TColor(5,1,1,0,"yellow");
      new TColor(6,1,0,1,"magenta");
      new TColor(7,0,1,1,"cyan");
      new TColor(10,0.999,0.999,0.999,"white");
      new TColor(11,0.754,0.715,0.676,"editcol");

      // The color white above is defined as being nearly white.
      // Sets the associated dark color also to white.
      //TColor *c110 = gROOT->GetColor(110);
      TColor::GetColorDark(10);
      TColor *c110 = gROOT->GetColor(110);
      c110->SetRGB(0.999,0.999,.999);

      // Initialize Custom colors
      new TColor(20,0.8,0.78,0.67);
      new TColor(31,0.54,0.66,0.63);
      new TColor(41,0.83,0.81,0.53);
      new TColor(30,0.52,0.76,0.64);
      new TColor(32,0.51,0.62,0.55);
      new TColor(24,0.70,0.65,0.59);
      new TColor(21,0.8,0.78,0.67);
      new TColor(47,0.67,0.56,0.58);
      new TColor(35,0.46,0.54,0.57);
      new TColor(33,0.68,0.74,0.78);
      new TColor(39,0.5,0.5,0.61);
      new TColor(37,0.43,0.48,0.52);
      new TColor(38,0.49,0.6,0.82);
      new TColor(36,0.41,0.51,0.59);
      new TColor(49,0.58,0.41,0.44);
      new TColor(43,0.74,0.62,0.51);
      new TColor(22,0.76,0.75,0.66);
      new TColor(45,0.75,0.51,0.47);
      new TColor(44,0.78,0.6,0.49);
      new TColor(26,0.68,0.6,0.55);
      new TColor(28,0.53,0.4,0.34);
      new TColor(25,0.72,0.64,0.61);
      new TColor(27,0.61,0.56,0.51);
      new TColor(23,0.73,0.71,0.64);
      new TColor(42,0.87,0.73,0.53);
      new TColor(46,0.81,0.37,0.38);
      new TColor(48,0.65,0.47,0.48);
      new TColor(34,0.48,0.56,0.6);
      new TColor(40,0.67,0.65,0.75);
      new TColor(29,0.69,0.81,0.78);

      // Initialize some additional greyish non saturated colors
      new TColor(8, 0.35,0.83,0.33);
      new TColor(9, 0.35,0.33,0.85);
      new TColor(12,.3,.3,.3,"grey12");
      new TColor(13,.4,.4,.4,"grey13");
      new TColor(14,.5,.5,.5,"grey14");
      new TColor(15,.6,.6,.6,"grey15");
      new TColor(16,.7,.7,.7,"grey16");
      new TColor(17,.8,.8,.8,"grey17");
      new TColor(18,.9,.9,.9,"grey18");
      new TColor(19,.95,.95,.95,"grey19");
      new TColor(50, 0.83,0.35,0.33);

      // Initialize the Pretty Palette Spectrum Violet->Red
      //   The color model used here is based on the HLS model which
      //   is much more suitable for creating palettes than RGB.
      //   Fixing the saturation and lightness we can scan through the
      //   spectrum of visible light by using "hue" alone.
      //   In Root hue takes values from 0 to 360.
      Float_t  saturation = 1;
      Float_t  lightness = 0.5;
      Float_t  maxHue = 280;
      Float_t  minHue = 0;
      Int_t    maxPretty = 50;
      Float_t  hue;

      for (i=0 ; i<maxPretty ; i++) {
         hue = maxHue-(i+1)*((maxHue-minHue)/maxPretty);
         TColor::HLStoRGB(hue, lightness, saturation, r, g, b);
         new TColor(i+51, r, g, b);
      }

      // Initialize special colors for x3d
      for (i = 1; i < 8; i++) {
         s0 = gROOT->GetColor(i);
         s0->GetRGB(r,g,b);
         if (i == 1) { r = 0.6; g = 0.6; b = 0.6; }
         if (r == 1) r = 0.9; if (r == 0) r = 0.1;
         if (g == 1) g = 0.9; if (g == 0) g = 0.1;
         if (b == 1) b = 0.9; if (b == 0) b = 0.1;
         TColor::RGBtoHLS(r,g,b,h,l,s);
         TColor::HLStoRGB(h,0.6*l,s,r,g,b);
         new TColor(200+4*i-3,r,g,b);
         TColor::HLStoRGB(h,0.8*l,s,r,g,b);
         new TColor(200+4*i-2,r,g,b);
         TColor::HLStoRGB(h,1.2*l,s,r,g,b);
         new TColor(200+4*i-1,r,g,b);
         TColor::HLStoRGB(h,1.4*l,s,r,g,b);
         new TColor(200+4*i  ,r,g,b);
      }
      
      // Create the ROOT Color Wheel
      TColor::CreateColorWheel();
   }
   SetPalette(0,0);
}

//______________________________________________________________________________
const char *TColor::AsHexString() const
{
   // Return color as hexidecimal string. This string can be directly passed
   // to, for example, TGClient::GetColorByName(). String will be reused so
   // copy immediately if needed.

   Int_t r, g, b, a;
   r = Int_t(fRed   * 255);
   g = Int_t(fGreen * 255);
   b = Int_t(fBlue  * 255);
   a = Int_t(fAlpha  * 255);

   return (a != 255) ? Form("#%02x%02x%02x%02x", a, r, g, b) : Form("#%02x%02x%02x", r, g, b);
}

//______________________________________________________________________________
void TColor::Copy(TObject &obj) const
{
   // Copy this color to obj.

   TNamed::Copy((TNamed&)obj);
   ((TColor&)obj).fRed   = fRed;
   ((TColor&)obj).fGreen = fGreen;
   ((TColor&)obj).fBlue  = fBlue;
   ((TColor&)obj).fHue   = fHue;
   ((TColor&)obj).fLight = fLight;
   ((TColor&)obj).fAlpha = fAlpha;
   ((TColor&)obj).fSaturation = fSaturation;
   ((TColor&)obj).fNumber = fNumber;
}


//______________________________________________________________________________
void TColor::CreateColorsGray()
{   
   // Create the Gray scale colors in the Color Wheel
   
   if (gROOT->GetColor(kGray)) return;
   TColor *gray  = new TColor(kGray,204./255.,204./255.,204./255.);
   TColor *gray1 = new TColor(kGray+1,153./255.,153./255.,153./255.);
   TColor *gray2 = new TColor(kGray+2,102./255.,102./255.,102./255.);
   TColor *gray3 = new TColor(kGray+3, 51./255., 51./255., 51./255.);
   gray ->SetName("kGray");
   gray1->SetName("kGray+1");
   gray2->SetName("kGray+2");
   gray3->SetName("kGray+3");
}

//______________________________________________________________________________
void TColor::CreateColorsCircle(Int_t offset, const char *name, UChar_t *rgb) 
{
   // Create the "circle" colors in the Color Wheel
   
   for (Int_t n=0;n<15;n++) {
      Int_t colorn = offset+n-10;
      TColor *color = gROOT->GetColor(colorn);
      if (!color) {
         color = new TColor(colorn,rgb[3*n]/255.,rgb[3*n+1]/255.,rgb[3*n+2]/255.);
         color->SetTitle(color->AsHexString());
         if      (n>10) color->SetName(Form("%s+%d",name,n-10));
         else if (n<10) color->SetName(Form("%s-%d",name,10-n));
         else           color->SetName(Form("%s",name));
      }
   }
}

//______________________________________________________________________________
void TColor::CreateColorsRectangle(Int_t offset, const char *name, UChar_t *rgb) 
{
   // Create the "rectangular" colors in the Color Wheel
   
   for (Int_t n=0;n<20;n++) {
      Int_t colorn = offset+n-9;
      TColor *color = gROOT->GetColor(colorn);
      if (!color) {
         color = new TColor(colorn,rgb[3*n]/255.,rgb[3*n+1]/255.,rgb[3*n+2]/255.);
         color->SetTitle(color->AsHexString());
         if      (n>9) color->SetName(Form("%s+%d",name,n-9));
         else if (n<9) color->SetName(Form("%s-%d",name,9-n));
         else          color->SetName(Form("%s",name));
      }
   }
}

//______________________________________________________________________________
void TColor::CreateColorWheel() 
{
   // Static function steering the creation of all colors in the ROOT Color Wheel
   
   UChar_t magenta[46]= {255,204,255
                        ,255,153,255, 204,153,204
                        ,255,102,255, 204,102,204, 153,102,153
                        ,255, 51,255, 204, 51,204, 153, 51,153, 102, 51,102
                        ,255,  0,255, 204,  0,204, 153,  0,153, 102,  0,102,  51,  0, 51};
                        
   UChar_t red[46]    = {255,204,204
                        ,255,153,153, 204,153,153
                        ,255,102,102, 204,102,102, 153,102,102
                        ,255, 51, 51, 204, 51, 51, 153, 51, 51, 102, 51, 51
                        ,255,  0,  0, 204,  0,  0, 153,  0,  0, 102,  0,  0,  51,  0,  0};
                        
   UChar_t yellow[46] = {255,255,204
                        ,255,255,153, 204,204,153
                        ,255,255,102, 204,204,102, 153,153,102
                        ,255,255, 51, 204,204, 51, 153,153, 51, 102,102, 51
                        ,255,255,  0, 204,204,  0, 153,153,  0, 102,102,  0,  51, 51,  0};
                        
   UChar_t green[46]  = {204,255,204
                        ,153,255,153, 153,204,153
                        ,102,255,102, 102,204,102, 102,153,102
                        , 51,255, 51,  51,204, 51,  51,153, 51,  51,102, 51
                        ,  0,255,  0,   0,204,  0,   0,153,  0,   0,102,  0,  0, 51,  0};
                        
   UChar_t cyan[46]   = {204,255,255
                        ,153,255,255, 153,204,204
                        ,102,255,255, 102,204,204, 102,153,153
                        , 51,255,255,  51,204,204,  51,153,153,  51,102,102
                        ,  0,255,255,   0,204,204,   0,153,153,   0,102,102,   0, 51,  51};
                        
   UChar_t blue[46]   = {204,204,255
                        ,153,153,255, 153,153,204
                        ,102,102,255, 102,102,204, 102,102,153
                        , 51, 51,255,  51, 51,204,  51, 51,153,  51, 51,102
                        ,  0,  0,255,   0,  0,204,   0,  0,153,   0,  0,102,   0,  0,  51};
                        
   UChar_t pink[60] = {255, 51,153,  204,  0,102,  102,  0, 51,  153,  0, 51,  204, 51,102
                      ,255,102,153,  255,  0,102,  255, 51,102,  204,  0, 51,  255,  0, 51
                      ,255,153,204,  204,102,153,  153, 51,102,  153,  0,102,  204, 51,153
                      ,255,102,204,  255,  0,153,  204,  0,153,  255, 51,204,  255,  0,153};
                      
   UChar_t orange[60]={255,204,153,  204,153,102,  153,102, 51,  153,102,  0,  204,153, 51
                      ,255,204,102,  255,153,  0,  255,204, 51,  204,153,  0,  255,204,  0
                      ,255,153, 51,  204,102,  0,  102, 51,  0,  153, 51,  0,  204,102, 51
                      ,255,153,102,  255,102,  0,  255,102, 51,  204, 51,  0,  255, 51,  0};
                       
   UChar_t spring[60]={153,255, 51,  102,204,  0,   51,102,  0,   51,153,  0,  102,204, 51
                      ,153,255,102,  102,255,  0,  102,255, 51,   51,204,  0,   51,255, 0
                      ,204,255,153,  153,204,102,  102,153, 51,  102,153,  0,  153,204, 51
                      ,204,255,102,  153,255,  0,  204,255, 51,  153,204,  0,  204,255,  0};

   UChar_t teal[60] = {153,255,204,  102,204,153,   51,153,102,    0,153,102,   51,204,153
                      ,102,255,204,    0,255,102,   51,255,204,    0,204,153,    0,255,204
                      , 51,255,153,    0,204,102,    0,102, 51,    0,153, 51,   51,204,102
                      ,102,255,153,    0,255,153,   51,255,102,    0,204, 51,    0,255, 51};

  UChar_t azure[60]  ={153,204,255,  102,153,204,   51,102,153,    0, 51,153,   51,102,204
                      ,102,153,255,    0,102,255,   51,102,255,    0, 51,204,    0, 51,255
                      , 51,153,255,    0,102,204,    0, 51,102,    0,102,153,   51,153,204
                      ,102,204,255,    0,153,255,   51,204,255,    0,153,204,    0,204,255};
      
  UChar_t violet[60] ={204,153,255,  153,102,204,  102, 51,153,  102,  0,153,  153, 51,204
                      ,204,102,255,  153,  0,255,  204, 51,255,  153,  0,204,  204,  0,255
                      ,153, 51,255,  102,  0,204,   51,  0,102,   51,  0,153,  102, 51,204
                      ,153,102,255,  102,  0,255,  102, 51,255,   51,  0,204,   51,  0,255};

   TColor::CreateColorsCircle(kMagenta,"kMagenta",magenta);
   TColor::CreateColorsCircle(kRed,    "kRed",    red);
   TColor::CreateColorsCircle(kYellow, "kYellow", yellow);
   TColor::CreateColorsCircle(kGreen,  "kGreen",  green);
   TColor::CreateColorsCircle(kCyan,   "kCyan",   cyan);
   TColor::CreateColorsCircle(kBlue,   "kBlue",   blue);
   
   TColor::CreateColorsRectangle(kPink,  "kPink",  pink);
   TColor::CreateColorsRectangle(kOrange,"kOrange",orange);
   TColor::CreateColorsRectangle(kSpring,"kSpring",spring);
   TColor::CreateColorsRectangle(kTeal,  "kTeal",  teal);
   TColor::CreateColorsRectangle(kAzure, "kAzure", azure);
   TColor::CreateColorsRectangle(kViolet,"kViolet",violet);
   
   TColor::CreateColorsGray();
}

//______________________________________________________________________________
Int_t TColor::GetColorPalette(Int_t i)
{
   // static function Return color number i in current palette.
   Int_t ncolors = fgPalette.fN;
   if (ncolors == 0) return 0;
   Int_t icol    = i%ncolors;
   if (icol < 0) icol = 0;
   return fgPalette.fArray[icol];
}

//______________________________________________________________________________
Int_t TColor::GetNumberOfColors()
{
   // static function: Return number of colors in the color palette
   
   return fgPalette.fN;
} 

//______________________________________________________________________________
ULong_t TColor::GetPixel() const
{
   // Return pixel value corresponding to this color. This pixel value can
   // be used in the GUI classes. This call does not work in batch mode since
   // it needs to communicate with the graphics system.

   if (gVirtualX && !gROOT->IsBatch()) {
      if (gApplication)
         gApplication->InitializeGraphics();
      return gVirtualX->GetPixel(fNumber);
   }

   return 0;
}

//______________________________________________________________________________
void TColor::HLS2RGB(Float_t hue, Float_t light, Float_t satur,
                     Float_t &r, Float_t &g, Float_t &b)
{
   // Static method to compute RGB from HLS. The l and s are between [0,1]
   // and h is between [0,360]. The returned r,g,b triplet is between [0,1].

   Float_t rh, rl, rs, rm1, rm2;
   rh = rl = rs = 0;
   if (hue   > 0) rh = hue;   if (rh > 360) rh = 360;
   if (light > 0) rl = light; if (rl > 1)   rl = 1;
   if (satur > 0) rs = satur; if (rs > 1)   rs = 1;

   if (rl <= 0.5)
      rm2 = rl*(1.0 + rs);
   else
      rm2 = rl + rs - rl*rs;
   rm1 = 2.0*rl - rm2;

   if (!rs) { r = rl; g = rl; b = rl; return; }
   r = HLStoRGB1(rm1, rm2, rh+120);
   g = HLStoRGB1(rm1, rm2, rh);
   b = HLStoRGB1(rm1, rm2, rh-120);
}

//______________________________________________________________________________
Float_t TColor::HLStoRGB1(Float_t rn1, Float_t rn2, Float_t huei)
{
   // Static method. Auxiliary to HLS2RGB().

   Float_t hue = huei;
   if (hue > 360) hue = hue - 360;
   if (hue < 0)   hue = hue + 360;
   if (hue < 60 ) return rn1 + (rn2-rn1)*hue/60;
   if (hue < 180) return rn2;
   if (hue < 240) return rn1 + (rn2-rn1)*(240-hue)/60;
   return rn1;
}

//______________________________________________________________________________
void TColor::HLS2RGB(Int_t h, Int_t l, Int_t s, Int_t &r, Int_t &g, Int_t &b)
{
   // Static method to compute RGB from HLS. The h,l,s are between [0,255].
   // The returned r,g,b triplet is between [0,255].

   Float_t hh, ll, ss, rr, gg, bb;

   hh = Float_t(h) * 360 / 255;
   ll = Float_t(l) / 255;
   ss = Float_t(s) / 255;

   TColor::HLStoRGB(hh, ll, ss, rr, gg, bb);

   r = (Int_t) (rr * 255);
   g = (Int_t) (gg * 255);
   b = (Int_t) (bb * 255);
}

//______________________________________________________________________________
void TColor::ls(Option_t *) const
{
   // List this color with its attributes.

   printf("Color:%d  Red=%f Green=%f Blue=%f Name=%s\n",
          fNumber, fRed, fGreen, fBlue, GetName());
}

//______________________________________________________________________________
void TColor::Print(Option_t *) const
{
   // Dump this color with its attributes.

   ls();
}

//______________________________________________________________________________
void TColor::RGB2HLS(Float_t rr, Float_t gg, Float_t bb,
                     Float_t &hue, Float_t &light, Float_t &satur)
{
   // Static method to compute HLS from RGB. The r,g,b triplet is between
   // [0,1], hue is between [0,360], light and satur are [0,1].

   Float_t rnorm, gnorm, bnorm, minval, maxval, msum, mdiff, r, g, b;
   r = g = b = 0;
   if (rr > 0) r = rr; if (r > 1) r = 1;
   if (gg > 0) g = gg; if (g > 1) g = 1;
   if (bb > 0) b = bb; if (b > 1) b = 1;

   minval = r;
   if (g < minval) minval = g;
   if (b < minval) minval = b;
   maxval = r;
   if (g > maxval) maxval = g;
   if (b > maxval) maxval = b;

   rnorm = gnorm = bnorm = 0;
   mdiff = maxval - minval;
   msum  = maxval + minval;
   light = 0.5 * msum;
   if (maxval != minval) {
      rnorm = (maxval - r)/mdiff;
      gnorm = (maxval - g)/mdiff;
      bnorm = (maxval - b)/mdiff;
   } else {
      satur = hue = 0;
      return;
   }

   if (light < 0.5)
      satur = mdiff/msum;
   else
      satur = mdiff/(2.0 - msum);

   if (r == maxval)
      hue = 60.0 * (6.0 + bnorm - gnorm);
   else if (g == maxval)
      hue = 60.0 * (2.0 + rnorm - bnorm);
   else
      hue = 60.0 * (4.0 + gnorm - rnorm);

   if (hue > 360)
      hue = hue - 360;
}

//______________________________________________________________________________
void TColor::RGB2HLS(Int_t r, Int_t g, Int_t b, Int_t &h, Int_t &l, Int_t &s)
{
   // Static method to compute HLS from RGB. The r,g,b triplet is between
   // [0,255], hue, light and satur are between [0,255].

   Float_t rr, gg, bb, hue, light, satur;

   rr = Float_t(r) / 255;
   gg = Float_t(g) / 255;
   bb = Float_t(b) / 255;

   TColor::RGBtoHLS(rr, gg, bb, hue, light, satur);

   h = (Int_t) (hue/360 * 255);
   l = (Int_t) (light * 255);
   s = (Int_t) (satur * 255);
}

//______________________________________________________________________________
void TColor::SetRGB(Float_t r, Float_t g, Float_t b)
{
   // Initialize this color and its associated colors.

   TColor::InitializeColors();
   fRed   = r;
   fGreen = g;
   fBlue  = b;

   if (fRed < 0) return;

   RGBtoHLS(r, g, b, fHue, fLight, fSaturation);

   Int_t nplanes = 16;
   if (gVirtualX) gVirtualX->GetPlanes(nplanes);
   if (nplanes == 0) nplanes = 16;

   // allocate color now (can be delayed when we have a large colormap)
#ifndef R__WIN32
   if (nplanes < 15)
#endif
      Allocate();

   if (fNumber > 50) return;

   // now define associated colors for WBOX shading
   Float_t dr, dg, db, lr, lg, lb;

   // set dark color
   HLStoRGB(fHue, 0.7*fLight, fSaturation, dr, dg, db);
   TColor *dark = gROOT->GetColor(100+fNumber);
   if (dark) {
      if (nplanes > 8) dark->SetRGB(dr, dg, db);
      else             dark->SetRGB(0.3,0.3,0.3);
   }

   // set light color
   HLStoRGB(fHue, 1.2*fLight, fSaturation, lr, lg, lb);
   TColor *light = gROOT->GetColor(150+fNumber);
   if (light) {
      if (nplanes > 8) light->SetRGB(lr, lg, lb);
      else             light->SetRGB(0.8,0.8,0.8);
   }
}

//______________________________________________________________________________
void TColor::Allocate()
{
   // Make this color known to the graphics system.

   if (gVirtualX && !gROOT->IsBatch())
      gVirtualX->SetRGB(fNumber, GetRed(), GetGreen(), GetBlue());
}

//______________________________________________________________________________
Int_t TColor::GetColor(const char *hexcolor)
{
   // Static method returning color number for color specified by
   // hex color string of form: #rrggbb, where rr, gg and bb are in
   // hex between [0,FF], e.g. "#c0c0c0".
   // If specified color does not exist it will be created with as
   // name "#rrggbb" with rr, gg and bb in hex between [0,FF].

   if (hexcolor && *hexcolor == '#') {
      Int_t r, g, b;
      if (sscanf(hexcolor+1, "%02x%02x%02x", &r, &g, &b) == 3)
         return GetColor(r, g, b);
   }
   ::Error("TColor::GetColor(const char*)", "incorrect color string");
   return 0;
}

//______________________________________________________________________________
Int_t TColor::GetColor(Float_t r, Float_t g, Float_t b)
{
   // Static method returning color number for color specified by
   // r, g and b. The r,g,b should be in the range [0,1].
   // If specified color does not exist it will be created
   // with as name "#rrggbb" with rr, gg and bb in hex between
   // [0,FF].

   Int_t rr, gg, bb;
   rr = Int_t(r * 255);
   gg = Int_t(g * 255);
   bb = Int_t(b * 255);

   return GetColor(rr, gg, bb);
}

//______________________________________________________________________________
Int_t TColor::GetColor(ULong_t pixel)
{
   // Static method returning color number for color specified by
   // system dependent pixel value. Pixel values can be obtained, e.g.,
   // from the GUI color picker.

   Int_t r, g, b;

   Pixel2RGB(pixel, r, g, b);

   return GetColor(r, g, b);
}

//______________________________________________________________________________
Int_t TColor::GetColor(Int_t r, Int_t g, Int_t b)
{
   // Static method returning color number for color specified by
   // r, g and b. The r,g,b should be in the range [0,255].
   // If the specified color does not exist it will be created
   // with as name "#rrggbb" with rr, gg and bb in hex between
   // [0,FF].

   TColor::InitializeColors();
   if (r < 0) r = 0;
   if (g < 0) g = 0;
   if (b < 0) b = 0;
   if (r > 255) r = 255;
   if (g > 255) g = 255;
   if (b > 255) b = 255;

   // Get list of all defined colors
   TObjArray *colors = (TObjArray*) gROOT->GetListOfColors();

   TColor *color = 0;

   // Look for color by name
   if ((color = (TColor*)colors->FindObject(Form("#%02x%02x%02x", r, g, b))))
      // We found the color by name, so we use that right away
      return color->GetNumber();

   Float_t rr, gg, bb;
   rr = Float_t(r)/255.;
   gg = Float_t(g)/255.;
   bb = Float_t(b)/255.;

   TIter next(colors);

   Int_t nplanes = 16;
   Float_t thres = 1.0/31.0;   // 5 bits per color : 0 - 0x1F !
   if (gVirtualX) gVirtualX->GetPlanes(nplanes);
   if (nplanes >= 24)
      thres = 1.0/255.0;       // 8 bits per color : 0 - 0xFF !

   // Loop over all defined colors
   while ((color = (TColor*)next())) {
      if (TMath::Abs(color->GetRed() - rr) > thres)
         continue;
      if (TMath::Abs(color->GetGreen() - gg) > thres)
         continue;
      if (TMath::Abs(color->GetBlue() - bb) > thres)
         continue;

      // We found a matching color in the color table
      return color->GetNumber();
   }

   // We didn't find a matching color in the color table, so we
   // add it. Note name is of the form "#rrggbb" where rr, etc. are
   // hexadecimal numbers.
   color = new TColor(colors->GetLast()+1, rr, gg, bb,
                      Form("#%02x%02x%02x", r, g, b));

   return color->GetNumber();
}

//______________________________________________________________________________
Int_t TColor::GetColorBright(Int_t n)
{
   // Static function: Returns the bright color number corresponding to n
   // If the TColor object does not exist, it is created.
   // The convention is that the bright color nb = n+150

   if (n < 0) return -1;
   
   // Get list of all defined colors
   TObjArray *colors = (TObjArray*) gROOT->GetListOfColors();
   Int_t ncolors = colors->GetSize();
   // Get existing color at index n
   TColor *color = 0;
   if (n < ncolors) color = (TColor*)colors->At(n);
   if (!color) return -1;
   
   //Get the rgb of the the new bright color corresponding to color n
   Float_t r,g,b;
   HLStoRGB(color->GetHue(), 1.2*color->GetLight(), color->GetSaturation(), r, g, b);
   
   //Build the bright color (unless the slot nb is already used)
   Int_t nb = n+150;
   TColor *colorb = 0;
   if (nb < ncolors) colorb = (TColor*)colors->At(nb);
   if (colorb) return nb;
   colorb = new TColor(nb,r,g,b);
   colorb->SetName(Form("%s_bright",color->GetName()));
   colors->AddAtAndExpand(colorb,nb);
   return nb;
}

//______________________________________________________________________________
Int_t TColor::GetColorDark(Int_t n)
{
   // Static function: Returns the dark color number corresponding to n
   // If the TColor object does not exist, it is created.
   // The convention is that the dark color nd = n+100

   if (n < 0) return -1;
   
   // Get list of all defined colors
   TObjArray *colors = (TObjArray*) gROOT->GetListOfColors();
   Int_t ncolors = colors->GetSize();
   // Get existing color at index n
   TColor *color = 0;
   if (n < ncolors) color = (TColor*)colors->At(n);
   if (!color) return -1;
   
   //Get the rgb of the the new dark color corresponding to color n
   Float_t r,g,b;
   HLStoRGB(color->GetHue(), 0.7*color->GetLight(), color->GetSaturation(), r, g, b);
   
   //Build the dark color (unless the slot nd is already used)
   Int_t nd = n+100;
   TColor *colord = 0;
   if (nd < ncolors) colord = (TColor*)colors->At(nd);
   if (colord) return nd;
   colord = new TColor(nd,r,g,b);
   colord->SetName(Form("%s_dark",color->GetName()));
   colors->AddAtAndExpand(colord,nd);
   return nd;
}
   

//______________________________________________________________________________
ULong_t TColor::Number2Pixel(Int_t ci)
{
   // Static method that given a color index number, returns the corresponding
   // pixel value. This pixel value can be used in the GUI classes. This call
   // does not work in batch mode since it needs to communicate with the
   // graphics system.


   TColor::InitializeColors();
   TColor *color = gROOT->GetColor(ci);
   if (color)
      return color->GetPixel();
   else
      ::Warning("TColor::Number2Pixel", "color with index %d not defined", ci);

   return 0;
}

//______________________________________________________________________________
ULong_t TColor::RGB2Pixel(Float_t r, Float_t g, Float_t b)
{
   // Convert r,g,b to graphics system dependent pixel value.
   // The r,g,b triplet must be [0,1].

   if (r < 0) r = 0;
   if (g < 0) g = 0;
   if (b < 0) b = 0;
   if (r > 1) r = 1;
   if (g > 1) g = 1;
   if (b > 1) b = 1;

   ColorStruct_t color;
   color.fRed   = UShort_t(r * 65535);
   color.fGreen = UShort_t(g * 65535);
   color.fBlue  = UShort_t(b * 65535);
   color.fMask  = kDoRed | kDoGreen | kDoBlue;
   gVirtualX->AllocColor(gVirtualX->GetColormap(), color);
   return color.fPixel;
}

//______________________________________________________________________________
ULong_t TColor::RGB2Pixel(Int_t r, Int_t g, Int_t b)
{
   // Convert r,g,b to graphics system dependent pixel value.
   // The r,g,b triplet must be [0,255].

   if (r < 0) r = 0;
   if (g < 0) g = 0;
   if (b < 0) b = 0;
   if (r > 255) r = 255;
   if (g > 255) g = 255;
   if (b > 255) b = 255;

   ColorStruct_t color;
   color.fRed   = UShort_t(r * 257);  // 65535/255
   color.fGreen = UShort_t(g * 257);
   color.fBlue  = UShort_t(b * 257);
   color.fMask  = kDoRed | kDoGreen | kDoBlue;
   gVirtualX->AllocColor(gVirtualX->GetColormap(), color);
   return color.fPixel;
}

//______________________________________________________________________________
void TColor::Pixel2RGB(ULong_t pixel, Float_t &r, Float_t &g, Float_t &b)
{
   // Convert machine dependent pixel value (obtained via RGB2Pixel or
   // via Number2Pixel() or via TColor::GetPixel()) to r,g,b triplet.
   // The r,g,b triplet will be [0,1].

   ColorStruct_t color;
   color.fPixel = pixel;
   gVirtualX->QueryColor(gVirtualX->GetColormap(), color);
   r = (Float_t)color.fRed / 65535;
   g = (Float_t)color.fGreen / 65535;
   b = (Float_t)color.fBlue / 65535;
}

//______________________________________________________________________________
void TColor::Pixel2RGB(ULong_t pixel, Int_t &r, Int_t &g, Int_t &b)
{
   // Convert machine dependent pixel value (obtained via RGB2Pixel or
   // via Number2Pixel() or via TColor::GetPixel()) to r,g,b triplet.
   // The r,g,b triplet will be [0,255].

   ColorStruct_t color;
   color.fPixel = pixel;
   gVirtualX->QueryColor(gVirtualX->GetColormap(), color);
   r = color.fRed / 257;
   g = color.fGreen / 257;
   b = color.fBlue / 257;
}

//______________________________________________________________________________
const char *TColor::PixelAsHexString(ULong_t pixel)
{
   // Convert machine dependent pixel value (obtained via RGB2Pixel or
   // via Number2Pixel() or via TColor::GetPixel()) to a hexidecimal string.
   // This string can be directly passed to, for example,
   // TGClient::GetColorByName(). String will be reused so copy immediately
   // if needed.

   Int_t r, g, b;
   Pixel2RGB(pixel, r, g, b);
   return Form("#%02x%02x%02x", r, g, b);
}

//______________________________________________________________________________
void TColor::SaveColor(ostream &out, Int_t ci)
{
    // Save a color with index > 228 as a C++ statement(s) on output stream out.

   char quote = '"';

   ULong_t pixel = Number2Pixel(ci);
   const char *cname = TColor::PixelAsHexString(pixel);

   if (gROOT->ClassSaved(TColor::Class())) {
      out << endl;
   } else {
      out << endl;
      out << "   Int_t ci;   // for color index setting" << endl;
   }

   out<<"   ci = TColor::GetColor("<<quote<<cname<<quote<<");"<<endl;
}

//______________________________________________________________________________
Bool_t TColor::IsGrayscale()
{
   // Return whether all colors return grayscale values
   return fgGrayscaleMode;
}

//______________________________________________________________________________
void TColor::SetGrayscale(Bool_t set /*= kTRUE*/)
{
   // Set whether all colors should return grayscale values
   if (fgGrayscaleMode == set) return;

   fgGrayscaleMode = set;

   if (!gVirtualX || gROOT->IsBatch()) return;

   TColor::InitializeColors();
   TIter iColor(gROOT->GetListOfColors());
   TColor* color = 0;
   while ((color = (TColor*) iColor()))
      color->Allocate();
}

//______________________________________________________________________________
Int_t TColor::CreateGradientColorTable(UInt_t Number, Double_t* Length,
                              Double_t* Red, Double_t* Green,
                              Double_t* Blue, UInt_t NColors)
{
  // STATIC function.
  // Linear gradient color table:
  // Red, Green and Blue are several RGB colors with values from 0.0 .. 1.0.
  // Their number is "Intervals".
  // Length is the length of the color interval between the RGB-colors:
  // Imaging the whole gradient goes from 0.0 for the first RGB color to 1.0
  // for the last RGB color, then each "Length"-entry in between stands for
  // the length of the intervall between the according RGB colors.
  //
  // This definition is similar to the povray-definition of gradient
  // color tables.
  //
  // In order to create a color table do the following:
  // Define the RGB Colors:
  // > UInt_t Number = 5;
  // > Double_t Red[5]   = { 0.00, 0.09, 0.18, 0.09, 0.00 };
  // > Double_t Green[5] = { 0.01, 0.02, 0.39, 0.68, 0.97 };
  // > Double_t Blue[5]  = { 0.17, 0.39, 0.62, 0.79, 0.97 };
  // Define the length of the (color)-interval between this points
  // > Double_t Stops[5] = { 0.00, 0.34, 0.61, 0.84, 1.00 };
  // i.e. the color interval between Color 2 and Color 3 is
  // 0.79 - 0.62 => 17 % of the total palette area between these colors
  //
  //  Original code by Andreas Zoglauer <zog@mpe.mpg.de>

   UInt_t g, c;
   UInt_t nPalette = 0;
   Int_t *palette = new Int_t[NColors+1];
   UInt_t nColorsGradient;
   TColor *color;
   Int_t highestIndex = 0;

   // Check if all RGB values are between 0.0 and 1.0 and
   // Length goes from 0.0 to 1.0 in increasing order.
   for (c = 0; c < Number; c++) {
      if (Red[c] < 0 || Red[c] > 1.0 ||
          Green[c] < 0 || Green[c] > 1.0 ||
          Blue[c] < 0 || Blue[c] > 1.0 ||
          Length[c] < 0 || Length[c] > 1.0) {
         //Error("CreateGradientColorTable",
         //      "All RGB colors and interval lengths have to be between 0.0 and 1.0");
         delete [] palette;
         return -1;
      }
      if (c >= 1) {
         if (Length[c-1] > Length[c]) {
            //Error("CreateGradientColorTable",
            //      "The interval lengths have to be in increasing order");
            delete [] palette;
            return -1;
         }
      }
   }

   // Search for the highest color index not used in ROOT:
   // We do not want to overwrite some colors...
   TSeqCollection *colorTable = gROOT->GetListOfColors();
   if ((color = (TColor *) colorTable->Last()) != 0) {
      if (color->GetNumber() > highestIndex) {
         highestIndex = color->GetNumber();
      }
      while ((color = (TColor *) (colorTable->Before(color))) != 0) {
         if (color->GetNumber() > highestIndex) {
            highestIndex = color->GetNumber();
         }
      }
   }
   highestIndex++;

   // Now create the colors and add them to the default palette:

   // For each defined gradient...
   for (g = 1; g < Number; g++) {
      // create the colors...
      nColorsGradient = (Int_t) (floor(NColors*Length[g]) - floor(NColors*Length[g-1]));
      for (c = 0; c < nColorsGradient; c++) {
         color = new TColor(highestIndex,
                            Red[g-1] + c * (Red[g] - Red[g-1])/ nColorsGradient,
                            Green[g-1] + c * (Green[g] - Green[g-1])/ nColorsGradient,
                            Blue[g-1] + c * (Blue[g] - Blue[g-1])/ nColorsGradient,
                            "  ");
         palette[nPalette] = highestIndex;
         nPalette++;
         highestIndex++;
      }
   }

   TColor::SetPalette(nPalette, palette);
   delete [] palette;

   return highestIndex - NColors;
}

//______________________________________________________________________________

//______________________________________________________________________________
void TColor::SetPalette(Int_t ncolors, Int_t *colors)
{
// static function
// The color palette is used by the histogram classes
//  (see TH1::Draw options).
// For example TH1::Draw("col") draws a 2-D histogram with cells
// represented by a box filled with a color CI function of the cell content.
// if the cell content is N, the color CI used will be the color number
// in colors[N],etc. If the maximum cell content is > ncolors, all
// cell contents are scaled to ncolors.
//
// if ncolors <= 0 a default palette (see below) of 50 colors is defined.
//     the colors defined in this palette are OK for coloring pads, labels
//
// if ncolors == 1 && colors == 0, then
//     a Pretty Palette with a Spectrum Violet->Red is created.
//   It is recommended to use this Pretty palette when drawing legos,
//   surfaces or contours.
//
// if ncolors > 50 and colors=0, the DeepSea palette is used.
//     (see TStyle::CreateGradientColorTable for more details)
//
// if ncolors > 0 and colors = 0, the default palette is used
// with a maximum of ncolors.
//
// The default palette defines:
//   index 0->9   : grey colors from light to dark grey
//   index 10->19 : "brown" colors
//   index 20->29 : "blueish" colors
//   index 30->39 : "redish" colors
//   index 40->49 : basic colors
//
//  The color numbers specified in the palette can be viewed by selecting
//  the item "colors" in the "VIEW" menu of the canvas toolbar.
//  The color parameters can be changed via TColor::SetRGB.

   Int_t i;
   static Int_t paletteType = 0;
   Int_t palette[50] = {19,18,17,16,15,14,13,12,11,20,
                        21,22,23,24,25,26,27,28,29,30, 8,
                        31,32,33,34,35,36,37,38,39,40, 9,
                        41,42,43,44,45,47,48,49,46,50, 2,
                         7, 6, 5, 4, 3, 112,1};
   // set default palette (pad type)
   if (ncolors <= 0) {
      ncolors = 50;
      fgPalette.Set(ncolors);
      for (i=0;i<ncolors;i++) fgPalette.fArray[i] = palette[i];
      paletteType = 1;
      return;
   }

   // set Pretty Palette Spectrum Violet->Red
   if (ncolors == 1 && colors == 0) {
      ncolors = 50;
      fgPalette.Set(ncolors);
      for (i=0;i<ncolors;i++) fgPalette.fArray[i] = 51+i;
      paletteType = 2;
      return;
   }

   // set DeepSea palette
   if (colors == 0 && ncolors > 50) {
      if (ncolors == fgPalette.fN && paletteType == 3) return;
      const Int_t nRGBs = 5;
      Double_t stops[nRGBs] = { 0.00, 0.34, 0.61, 0.84, 1.00 };
      Double_t red[nRGBs] = { 0.00, 0.09, 0.18, 0.09, 0.00 };
      Double_t green[nRGBs] = { 0.01, 0.02, 0.39, 0.68, 0.97 };
      Double_t blue[nRGBs] = { 0.17, 0.39, 0.62, 0.79, 0.97 };
      TColor::CreateGradientColorTable(nRGBs, stops, red, green, blue, ncolors);
      paletteType = 3;
      return;
   }

   // set user defined palette
   fgPalette.Set(ncolors);
   if (colors)  for (i=0;i<ncolors;i++) fgPalette.fArray[i] = colors[i];
   else         for (i=0;i<ncolors;i++) fgPalette.fArray[i] = palette[i];
   paletteType = 4;
}

