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

   R__ATTR_CLASS(RColor, "color_", AddString("hex", "").AddString("rgb", "").AddString("name", "").AddString("a", ""));

   using RGB_t = std::array<int, 3>;

protected:

   std::string toHex(int v) const
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

public:

   RColor(int r, int g, int b) : RColor() { SetHex(r, g, b); }

   RColor(int r, int g, int b, double alfa) : RColor()
   {
      SetHex(r, g, b);
      SetAlpha(alfa);
   }

   RColor(const RGB_t &rgb) : RColor() { SetHex(rgb[0], rgb[1], rgb[2]); }

   std::string GetRGB() const { return GetValue<std::string>("rgb"); }

   RColor &SetRGB(int r, int g, int b)
   {
      return SetRGB(std::to_string(r) + "," + std::to_string(g) + "," + std::to_string(b));
   }

   RColor &SetRGB(const std::string &_rgb)
   {
      ClearValue("hex");
      ClearValue("name");
      SetValue("rgb", _rgb);
      return *this;
   }

   /** Set r/g/b/ components of color as hex code, default for the color */
   RColor &SetHex(int r, int g, int b)
   {
      return SetHex(toHex(r) + toHex(g) + toHex(b));
   }

   /** Set color as hex string like 00FF00 */
   RColor &SetHex(const std::string &_hex)
   {
      ClearValue("rgb");
      ClearValue("name");
      SetValue("hex", _hex);
      return *this;
   }

   std::string GetHex() const { return GetValue<std::string>("hex"); }

   RColor &SetName(const std::string &_name)
   {
      ClearValue("hex");
      ClearValue("rgb");
      SetValue("name", _name);
      return *this;
   }

   std::string GetName() const { return GetValue<std::string>("name"); }


   double GetAlpha() const
   {
      auto hex = GetAlphaHex();
      if (hex.empty())
         return 1.;
      return std::strtol(hex.c_str(), nullptr, 16) / 255.;
   }

   std::string GetAlphaHex() const { return GetValue<std::string>("a"); }

   bool HasAlpha() const { return HasValue("a"); }

   RColor &SetAlpha(double _alfa)
   {
      return SetAlphaHex(toHex((int) (_alfa*255)));
   }

   RColor &SetAlphaHex(const std::string &_alfa)
   {
      SetValue("a", _alfa);
      return *this;
   }

   std::string AsSVG() const
   {
      bool has_alpha = HasAlpha();

      auto hex = GetHex();
      if (!hex.empty()) {
         std::string res = "#";
         res.append(hex);
         if (has_alpha) res.append(GetAlphaHex());
         return res;
      }

      auto rgb = GetRGB();
      if (!rgb.empty()) {
         if (!has_alpha) return std::string("rgb(") + rgb + ")";
         char alphabuf[10];
         snprintf(alphabuf, sizeof(alphabuf), "%5.3f", GetAlpha());
         return std::string("rgba(") + rgb + "," + alphabuf + ")";
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
      return (lhs.GetHex() == rhs.GetHex()) && (lhs.GetName() == rhs.GetName()) && (lhs.GetRGB() == rhs.GetRGB()) &&
             (lhs.GetAlphaHex() == rhs.GetAlphaHex());
   }
};

} // namespace Experimental
} // namespace ROOT

#endif
