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

#include <ROOT/TLogger.hxx>

#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <sstream>
#include <unordered_map>

using namespace ROOT::Experimental;

// RColor constexpr values:
constexpr RColor::Alpha RColor::kOpaque;
constexpr RColor::Alpha RColor::kTransparent;
constexpr RColor::RGBA RColor::kRed;
constexpr RColor::RGBA RColor::kGreen;
constexpr RColor::RGBA RColor::kBlue;
constexpr RColor::RGBA RColor::kWhite;
constexpr RColor::RGBA RColor::kBlack;
constexpr RColor::RGBA RColor::kInvisible;
constexpr RColor::AutoTag RColor::kAuto;


float RColor::GetPaletteOrdinal() const
{
   if (fKind != EKind::kPalettePos)
      throw std::runtime_error("This color does not represent a palette ordinal!");
   return fRedOrPalettePos;
}

bool RColor::AssertNotPalettePos() const
{
   if (fKind == EKind::kPalettePos) {
      throw std::runtime_error("This color does not represent a palette ordinal!");
      return false;
   }
   return true;
}

namespace {
   static RColor ParseRGBToColor(const std::string &strval, const std::string &name)
   {
      auto rgbalen = strval.length() - 1;
      if (rgbalen != 3 && rgbalen != 4 && rgbalen != 6 && rgbalen != 8) {
         R__ERROR_HERE("Graf2d") << "Invalid value for RColor default style " << name
            << " with value \"" << strval
            << "\": expect '#' followed by 3, 4, 6 or 8 hex digits (#rgb, #rgba, #rrggbbaa or #rrggbb).";
         return RColor::kBlack;
      }
      std::size_t pos;
      long long rgbaLL = std::stoll(strval.substr(1), &pos, /*base*/ 16);
      if (pos != 3 && pos != 4 && pos != 6 && pos != 8) {
         R__ERROR_HERE("Graf2d") << "Invalid value while parsing default style value for RColor " << name
            << " with value \"" << strval
            << "\": expect '#' followed by 3, 4, 6 or 8 hex digits (#rgb, #rgba, #rrggbbaa or #rrggbb).";
         return RColor::kBlack;
      }
      if (pos != rgbalen) {
         R__WARNING_HERE("Graf2d") << "Leftover characters while parsing default style value for RColor " << name
            << " with value \"" << strval << "\", remainder: \"" << strval.substr(pos - 1) << "\"";
         return RColor::kBlack;
      }
      std::array<float, 4> rgba = {0., 0., 0., 1.};
      const bool haveAlpha = pos == 4 || pos == 8;
      // #rrggbb[aa] has 8 bits per channel, #rgb[a] has 4.
      const int bitsPerChannel = (pos > 4) ? 8 : 4;
      const int bitMask = (1 << bitsPerChannel) - 1;
      const float bitMaskFloat = static_cast<float>(bitMask);
      for (int i = haveAlpha ? 3 : 2; i >= 0; --i) {
         rgba[i] = (rgbaLL & bitMask) / bitMaskFloat;
         rgbaLL >>= bitsPerChannel;
      }
      return RColor(rgba);
   }

   RColor ParseColorNameToColor(const std::string &strval, const std::string &name)
   {
      std::string nameLow = strval;
      std::transform(nameLow.begin(), nameLow.end(), nameLow.begin(),
         // tolower has undef behavior for char argument; cast it.
         // And must not take &stdlib function, so lambda it is.
         [](char c) { return std::tolower(static_cast<unsigned char>(c)); });
      using namespace std::string_literals;
      static const std::unordered_map<std::string, RColor> mapNamesToColor {
         {"red"s, RColor{RColor::kRed}},
         {"green"s, RColor{RColor::kGreen}},
         {"blue"s, RColor{RColor::kBlue}},
         {"white"s, RColor{RColor::kWhite}},
         {"black"s, RColor{RColor::kBlack}},
         {"invisible"s, RColor{RColor::kInvisible}},
         {"auto"s, RColor{RColor::kAuto}}
      };
      auto itMap = mapNamesToColor.find(nameLow);
      if (itMap == mapNamesToColor.end()) {
         R__WARNING_HERE("Graf2d") << "Cannot parse RColor " << name
            << " with value \"" << strval << "\": unknown color name.";
         return RColor::kBlack;
      }
      return itMap->second;
   }
} // unnamed namespace

////////////////////////////////////////////////////////////////////////////////
/// Initialize a RColor from a string value.
/// Colors can be specified as RGBA (red green blue alpha) or RRGGBBAA:
///     %fa7f %ffa07bff # hash introduces a comment!
/// For all predefined colors in RColor, colors can be specified as name without leading 'k', e.g. `red` for
/// `RColor::kRed`.
/// Prints an error and returns `RColor::kBlack` if the attribute string cannot be parsed or if the attribute has no
/// entry in `fAttrs`.
///
///\param name - the attribute name (for diagnostic purposes).
///\param strval - the attribute value as a string.
ROOT::Experimental::RColor ROOT::Experimental::FromAttributeString(const std::string &strval, const std::string &name, RColor*)
{
   if (strval[0] == '#') {
      return ParseRGBToColor(strval, name);
   }
   return ParseColorNameToColor(strval, name);
}

////////////////////////////////////////////////////////////////////////////////
/// Return a `std::string` representation of a RColor, suitable as input to ColorFromString().
///
///\param val - the color to be "stringified".
std::string ROOT::Experimental::ToAttributeString(const RColor& val)
{
   // For now, always create "#RRGGBBAA".
   std::string ret("#");
   ret.reserve(9);
   std::array<float, 4> arr{ val.GetRed(), val.GetGreen(), val.GetBlue(), val.GetAlpha() };
   auto toHex = [](int i) { return  i > 9 ? 'A' - 10 + i : '0' + i; };
   for (auto &component: arr) {
      int comp8Bit =  static_cast<int>(std::round(255 * component));
      ret.push_back(toHex(comp8Bit / 16));
      ret.push_back(toHex(comp8Bit % 16));
   }
   return ret;
}
