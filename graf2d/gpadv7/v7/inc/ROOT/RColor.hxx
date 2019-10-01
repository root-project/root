/// \file ROOT/RColor.hxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2017-09-26
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RColor
#define ROOT7_RColor

#include <ROOT/RAttrBase.hxx>

#include <array>


namespace ROOT {
namespace Experimental {

// TODO: see also imagemagick's C++ interface for RColor operations!
// https://www.imagemagick.org/api/magick++-classes.php

class RColor : public RAttrBase {

   R__ATTR_CLASS(RColor, "color_", AddString("rgb", "").AddString("a", "").AddString("name", ""));

   using RGB_t = std::array<int, 3>;

private:

   static std::string toHex(int v);

   /** Set RGB values as floats, each from 0..1. Real color values will be stored in hex format */
   RColor &SetRGBFloat(float r, float g, float b)
   {
      return SetRGB(int(r*255),int(g*255),int(b*255));
   }

   bool GetRGBFloat(float &r, float &g, float &b) const;

   int GetColorComponent(int indx) const;

public:

   /** Construct color with provided r,g,b values */
   RColor(int r, int g, int b) : RColor() { SetRGB(r, g, b); }

   /** Construct color with provided r,g,b and alpha values */
   RColor(int r, int g, int b, float alpha) : RColor()
   {
      SetRGB(r, g, b);
      SetAlpha(alpha);
   }

   /** Construct color with provided RGB_t value */
   RColor(const RGB_t &rgb) : RColor() { SetRGB(rgb[0], rgb[1], rgb[2]); }

   /** Set r/g/b/ components of color as hex code, default for the color */
   RColor &SetRGB(int r, int g, int b)
   {
      return SetHex(toHex(r) + toHex(g) + toHex(b));
   }

   /** Set color as hex string like 00FF00 */
   RColor &SetHex(const std::string &_hex)
   {
      SetValue("rgb", _hex);
      return *this;
   }

   /** Return color as hex string like 00FF00 */
   std::string GetHex() const { return GetValue<std::string>("rgb"); }

   bool GetRGB(int &r, int &g, int &b) const;

   /** Returns red color component 0..255 */
   int GetRed() const { return GetColorComponent(0); }

   /** Returns green color component 0..255 */
   int GetGreen() const { return GetColorComponent(1); }

   /** Returns blue color component 0..255 */
   int GetBlue() const { return GetColorComponent(2); }

   /** Clear RGB color value (if any) */
   void ClearRGB()
   {
      ClearValue("rgb");
   }

   /** Set color as plain SVG name like "white" or "lightblue". Clears RGB component before */
   RColor &SetName(const std::string &_name)
   {
      ClearRGB();
      SetValue("name", _name);
      return *this;
   }

   /** Returns color as plain SVG name like "white" or "lightblue" */
   std::string GetName() const { return GetValue<std::string>("name"); }

   /** Clear color plain SVG name (if any) */
   void ClearName()
   {
      ClearValue("name");
   }

   /** Returns color alpha (opacity) as float from 0. to 1. */
   float GetAlpha() const
   {
      auto hex = GetAlphaHex();
      if (hex.empty())
         return 1.;
      return std::strtol(hex.c_str(), nullptr, 16) / 255.;
   }

   /** Returns color alpha (opacity) as hex string like FF. Default is empty */
   std::string GetAlphaHex() const { return GetValue<std::string>("a"); }

   /** Returns true if color alpha (opacity) was specified */
   bool HasAlpha() const { return HasValue("a"); }

   /** Set color alpha (opacity) value - from 0 to 1 */
   RColor &SetAlpha(float _alpha)
   {
      return SetAlphaHex(toHex((int) (_alpha*255)));
   }

   /** Set color alpha (opacity) value as hex string */
   RColor &SetAlphaHex(const std::string &_alfa)
   {
      SetValue("a", _alfa);
      return *this;
   }

   /** Return the Hue, Light, Saturation (HLS) definition of this RColor */
   bool GetHLS(float &hue, float &light, float &satur) const;

   /** Set the Red Green and Blue (RGB) values from the Hue, Light, Saturation (HLS). */
   RColor &SetHLS(float hue, float light, float satur);

   /** Returns color value as it will be used in SVG drawing
    * It either include hex format #66FF66 or just plain SVG name */
   std::string AsSVG() const
   {
      auto hex = GetHex();
      if (!hex.empty()) {
         std::string res = "#";
         res.append(hex);
         res.append(GetAlphaHex());
         return res;
      }

      // check that alpha is not specified
      return GetName();
   }

   static constexpr RGB_t kRed{{255, 0, 0}};
   static constexpr RGB_t kGreen{{0, 255, 0}};
   static constexpr RGB_t kBlue{{0, 0, 255}};
   static constexpr RGB_t kWhite{{255, 255, 255}};
   static constexpr RGB_t kBlack{{0, 0, 0}};
   static constexpr double kTransparent{0.};
   static constexpr double kOpaque{1.};

   friend bool operator==(const RColor &lhs, const RColor &rhs)
   {
      return (lhs.GetHex() == rhs.GetHex()) && (lhs.GetName() == rhs.GetName()) &&
             (lhs.GetAlphaHex() == rhs.GetAlphaHex());
   }
};

} // namespace Experimental
} // namespace ROOT

#endif
