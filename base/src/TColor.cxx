// @(#)root/base:$Name:  $:$Id: TColor.cxx,v 1.8 2001/10/24 13:16:56 brun Exp $
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "IOStream.h"
#include "TROOT.h"
#include "TColor.h"
#include "TObjArray.h"
#include "TVirtualPad.h"
#include "TVirtualX.h"
#include "TError.h"


ClassImp(TColor)

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
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
TColor::TColor(): TNamed()
{
   // Default ctor.

   fNumber = -1;
   fRed = fGreen = fBlue = fHue = fLight = fSaturation = -1;
}

//______________________________________________________________________________
TColor::TColor(Int_t color, Float_t r, Float_t g, Float_t b, const char *name)
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
}

//______________________________________________________________________________
TColor::~TColor()
{
   // Color destructor.

   gROOT->GetListOfColors()->Remove(this);
}

//______________________________________________________________________________
TColor::TColor(const TColor &color)
{
   // Color copy ctor.

   ((TColor&)color).Copy(*this);
}

//______________________________________________________________________________
void TColor::Copy(TObject &obj)
{
   // Copy this color to obj.

   TNamed::Copy((TNamed&)obj);
   ((TColor&)obj).fRed   = fRed;
   ((TColor&)obj).fGreen = fGreen;
   ((TColor&)obj).fBlue  = fBlue;
   ((TColor&)obj).fHue   = fHue;
   ((TColor&)obj).fLight = fLight;
   ((TColor&)obj).fSaturation = fSaturation;
}

//______________________________________________________________________________
void TColor::HLStoRGB(Float_t hue, Float_t light, Float_t satur, Float_t &r, Float_t &g, Float_t &b)
{
   // Static method to compute HLS from RGB (see HIGZ routine IGHTOR).

   Float_t rh, rl, rs, rm1, rm2;
   rh = rl = rs = 0;
   if (hue   > 0) rh = hue;   if (rh > 360) rh = 360;
   if (light > 0) rl = light; if (rl > 1)   rl = 1;
   if (satur > 0) rs = satur; if (rs > 1)   rs = 1;

   if (rl <= 0.5) rm2 = rl*(1+rs);
   else           rm2 = rl + rs - rl*rs;
   rm1 = 2*rl - rm2;

   if (!rs) { r = rl; g = rl; b = rl; return;}
   r = HLStoRGB1(rm1, rm2, rh+120);
   g = HLStoRGB1(rm1, rm2, rh);
   b = HLStoRGB1(rm1, rm2, rh-120);
}

//______________________________________________________________________________
Float_t TColor::HLStoRGB1(Float_t rn1, Float_t rn2, Float_t huei)
{
   // Static method. Auxiliary to HLStoRGB (see HIGZ routine IGHR01).

   Float_t hue = huei;
   if (hue > 360) hue = hue - 360;
   if (hue < 0)   hue = hue + 360;
   if (hue < 60 ) return rn1 + (rn2-rn1)*hue/60;
   if (hue < 180) return rn2;
   if (hue < 240) return rn1 + (rn2-rn1)*(240-hue)/60;
   return rn1;
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
void TColor::RGBtoHLS(Float_t r, Float_t g, Float_t b, Float_t &hue, Float_t &light, Float_t &satur )
{
   // Static method to compute HLS from RGB.

   Float_t rnorm, gnorm, bnorm, minval, maxval, msum, mdiff;
   minval = r;
   if (g <minval) minval = g;
   if (b <minval) minval = b;
   maxval = r;
   if (g >maxval) maxval = g;
   if (b >maxval) maxval = b;

   rnorm  = gnorm = bnorm = 0;
   mdiff  = maxval - minval;
   msum   = maxval + minval;
   light = 0.5*msum;
   if (maxval != minval) {
      rnorm = (maxval - r)/mdiff;
      gnorm = (maxval - g)/mdiff;
      bnorm = (maxval - b)/mdiff;
   }
   else { satur = hue = 0; return;}

   if (light <= 0.5) satur = mdiff/msum;
   else satur = mdiff/(2-msum);

   if (r == maxval) hue = 60.0 * (6.0 + bnorm - gnorm);
   if (g == maxval) hue = 60.0 * (2.0 + rnorm - bnorm);
   if (b == maxval) hue = 60.0 * (4.0 + gnorm - rnorm);
   if (hue > 360) hue = hue -360;
}

//______________________________________________________________________________
void TColor::SetRGB(Float_t r, Float_t g, Float_t b)
{
   // Initialize this color and its assosiated colors.

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
      gVirtualX->SetRGB(fNumber, fRed, fGreen, fBlue);
}

//______________________________________________________________________________
Int_t TColor::GetColor(const char *hexcolor)
{
   // Static method returning color number for color specified by
   // hex color string of form: #rrggbb, where rr, gg and bb are in
   // hex between [0,FF], e.g. "#c0c0c0".
   // If color specified color does not exist it will be created
   // with as name "#rrggbb".

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
   // If color specified color does not exist it will be created
   // with as name "#rrggbb" with rr, gg and bb are in hex between
   // [0,255].

   Int_t rr, gg, bb;
   rr = Int_t(r * 255);
   gg = Int_t(g * 255);
   bb = Int_t(b * 255);

   return GetColor(rr, gg, bb);
}

//______________________________________________________________________________
Int_t TColor::GetColor(Int_t r, Int_t g, Int_t b)
{
   // Static method returning color number for color specified by
   // r, g and b. The r,g,b should be in the range [0,255].
   // If color specified color does not exist it will be created
   // with as name "#rrggbb" with rr, gg and bb are in hex between.

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

   // Loop over all defined colors
   while ((color = (TColor*)next())) {
      if (color->GetRed() != rr)
         continue;
      if (color->GetGreen() != gg)
         continue;
      if (color->GetBlue() != bb)
         continue;

      // We found a matching color in the color table
      return color->GetNumber();
   }

   // We didn't find a matching color in the color table, so we
   // add it. Note name is of the form "#rrggbb" where rr, etc. are
   // hexadecimal numbers.
   color = new TColor(colors->GetEntries(), rr, gg, bb,
                      Form("#%02x%02x%02x", r, g, b));

   return color->GetNumber();
}


