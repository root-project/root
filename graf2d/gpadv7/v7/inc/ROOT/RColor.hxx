/// \file ROOT/RColorOld.hxx
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

#ifndef ROOT7_RColorOld
#define ROOT7_RColorOld

#include <array>
#include <vector>
#include <string>

#include <ROOT/RDrawingAttr.hxx>

namespace ROOT {
namespace Experimental {

// TODO: see also imagemagick's C++ interface for RColorOld operations!
// https://www.imagemagick.org/api/magick++-classes.php

class RColor : public RAttributesVisitor {

protected:
   const RDrawableAttributes::Map_t &GetDefaults() const override
   {
      static auto dflts = RDrawableAttributes::Map_t().AddString("rgb","0,0,0").AddDouble("a",1.);
      return dflts;
   }

public:

   using RGB_t = std::array<int, 3>;


   using RAttributesVisitor::RAttributesVisitor;

   RColor(int r, int g, int b) : RColor()
   {
      SetRGB(r,g,b);
   }

   RColor(int r, int g, int b, double alfa) : RColor()
   {
      SetRGB(r,g,b);
      SetAlpha(alfa);
   }

   RColor(const RGB_t &rgb) : RColor()
   {
      SetRGB(rgb[0],rgb[1],rgb[2]);
   }


   std::string GetRGB() const { return GetValue<std::string>("rgb"); }
   RColor &SetRGB(const std::string &_rgb) { SetValue("rgb", _rgb); return *this; }
   RColor &SetRGB(int r, int g, int b) { return SetRGB(std::to_string(r) + "," + std::to_string(g) + "," + std::to_string(b)); }

   double GetAlpha() const { return GetValue<double>("a"); }
   bool HasAlpha() const { return HasValue("a"); }
   RColor &SetAlpha(double _alfa) { SetValue("a", _alfa); return *this; }

   std::string AsSVG() const
   {
      auto rgb = GetRGB();
      if (HasAlpha())
         return std::string("rgba(") + rgb + "," + std::to_string(GetAlpha()) + ")";
       return std::string("rgb(") + rgb + ")";
   }

   static constexpr RGB_t kRed{{255, 0, 0}};
   static constexpr RGB_t kGreen{{0, 255, 0}};
   static constexpr RGB_t kBlue{{0, 0, 255}};
   static constexpr RGB_t kWhite{{255, 255, 255}};
   static constexpr RGB_t kBlack{{0, 0, 0}};
   static constexpr double kTransparent{0.};
   static constexpr double kOpaque{1.};


   friend bool operator==(const RColor& lhs, const RColor& rhs){ return (lhs.GetRGB() == rhs.GetRGB()) && (lhs.HasAlpha() == rhs.HasAlpha()) && (lhs.GetAlpha() == rhs.GetAlpha()); }


};


} // namespace Experimental
} // namespace ROOT

#endif
