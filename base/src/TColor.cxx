// @(#)root/base:$Name:  $:$Id: TColor.cxx,v 1.26 2007/01/12 16:03:15 brun Exp $
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


ClassImp(TColor)

Bool_t TColor::fgGrayscaleMode = kFALSE;

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
   const char *cname = GetName();

   // enter in the list of colors
   TObjArray *lcolors = (TObjArray*)gROOT->GetListOfColors();
   lcolors->AddAtAndExpand(this, color);

   if (color > 0 && color < 51) {
      // now create associated colors for WBOX shading
      sprintf(aname,"%s%s",cname,"_dark");
      new TColor(100+color, -1, -1, -1, aname);
      sprintf(aname,"%s%s",cname,"_bright");
      new TColor(150+color, -1, -1, -1, aname);
   }

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
ULong_t TColor::GetPixel() const
{
   // Return pixel value corresponding to this color. This pixel value can
   // be used in the GUI classes. This call does not work in batch mode since
   // it needs to communicate with the graphics system.

   if (gVirtualX && !gROOT->IsBatch())
      return gVirtualX->GetPixel(fNumber);

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
ULong_t TColor::Number2Pixel(Int_t ci)
{
   // Static method that given a color index number, returns the corresponding
   // pixel value. This pixel value can be used in the GUI classes. This call
   // does not work in batch mode since it needs to communicate with the
   // graphics system.

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

   TIter iColor(gROOT->GetListOfColors());
   TColor* color = 0;
   while ((color = (TColor*) iColor()))
      color->Allocate();
}
