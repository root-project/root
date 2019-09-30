/// \file RColor.cxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2017-09-27
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RColor.hxx"

using namespace ROOT::Experimental;

constexpr RColor::RGB_t RColor::kRed;
constexpr RColor::RGB_t RColor::kGreen;
constexpr RColor::RGB_t RColor::kBlue;
constexpr RColor::RGB_t RColor::kWhite;
constexpr RColor::RGB_t RColor::kBlack;
constexpr double RColor::kTransparent;
constexpr double RColor::kOpaque;


std::string RColor::toHex(int v)
{
   static const char *digits = "0123456789ABCDEF";
   if (v < 0)
      v = 0;
   else if (v > 255)
      v = 255;

   std::string res(2,'0');
   res[0] = digits[v >> 4];
   res[1] = digits[v & 0xf];
   return res;
}

bool RColor::GetRGB(int &r, int &g, int &b) const
{
   auto hex = GetHex();
   if (hex.length() != 6)
      return false;

   r = std::stoi(hex.substr(0,2), nullptr, 16);
   g = std::stoi(hex.substr(2,2), nullptr, 16);
   b = std::stoi(hex.substr(4,2), nullptr, 16);
   return true;
}

bool RColor::GetRGBFloat(float &r, float &g, float &b) const
{
   int ri, gi, bi;
   if (!GetRGB(ri,gi,bi))
      return false;
   r = ri/255.;
   g = gi/255.;
   b = bi/255.;
   return true;
}

/// Return the Hue, Light, Saturation (HLS) definition of this RColor
bool RColor::GetHLS(float &hue, float &light, float &satur) const
{
   float red, green, blue;
   if (!GetRGBFloat(red,green,blue))
      return false;

   hue = light = satur = 0.;

   float rnorm, gnorm, bnorm, minval, maxval, msum, mdiff;
   minval = maxval =0 ;

   minval = red;
   if (green < minval) minval = green;
   if (blue < minval)  minval = blue;
   maxval = red;
   if (green > maxval) maxval = green;
   if (blue > maxval)  maxval = blue;

   rnorm = gnorm = bnorm = 0;
   mdiff = maxval - minval;
   msum  = maxval + minval;
   light = 0.5 * msum;
   if (maxval != minval) {
      rnorm = (maxval - red)/mdiff;
      gnorm = (maxval - green)/mdiff;
      bnorm = (maxval - blue)/mdiff;
   } else {
      satur = hue = 0;
      return true;
   }

   if (light < 0.5) satur = mdiff/msum;
   else             satur = mdiff/(2.0 - msum);

   if      (red == maxval) hue = 60.0 * (6.0 + bnorm - gnorm);
   else if (green == maxval)           hue = 60.0 * (2.0 + rnorm - bnorm);
   else                                hue = 60.0 * (4.0 + gnorm - rnorm);

   if (hue > 360) hue = hue - 360;
   return true;
}

/// Set the Red Green and Blue (RGB) values from the Hue, Light, Saturation (HLS).
RColor &RColor::SetHLS(float hue, float light, float satur)
{
   float rh, rl, rs, rm1, rm2;
   rh = rl = rs = 0;
   if (hue   > 0) { rh = hue;   if (rh > 360) rh = 360; }
   if (light > 0) { rl = light; if (rl > 1)   rl = 1; }
   if (satur > 0) { rs = satur; if (rs > 1)   rs = 1; }

   if (rl <= 0.5) rm2 = rl*(1.0 + rs);
   else           rm2 = rl + rs - rl*rs;
   rm1 = 2.0*rl - rm2;

   if (!rs) { SetRGBFloat(rl, rl, rl); return *this; }

   auto toRGB = [rm1, rm2] (float h) {
      if (h > 360) h = h - 360;
      if (h < 0)   h = h + 360;
      if (h < 60 ) return rm1 + (rm2-rm1)*h/60;
      if (h < 180) return rm2;
      if (h < 240) return rm1 + (rm2-rm1)*(240-h)/60;
      return rm1;
   };

   return SetRGBFloat(toRGB(rh+120), toRGB(rh), toRGB(rh-120));
}
