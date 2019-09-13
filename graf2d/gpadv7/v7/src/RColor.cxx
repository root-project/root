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

#include <ROOT/RLogger.hxx>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <exception>
#include <iomanip>
#include <sstream>
#include <unordered_map>

using namespace ROOT::Experimental;

// RColorOld constexpr values:
constexpr RColorOld::Alpha RColorOld::kOpaque;
constexpr RColorOld::Alpha RColorOld::kTransparent;
constexpr RColorOld::RGBA RColorOld::kRed;
constexpr RColorOld::RGBA RColorOld::kGreen;
constexpr RColorOld::RGBA RColorOld::kBlue;
constexpr RColorOld::RGBA RColorOld::kWhite;
constexpr RColorOld::RGBA RColorOld::kBlack;
constexpr RColorOld::RGBA RColorOld::kInvisible;
constexpr RColorOld::AutoTag RColorOld::kAuto;


float RColorOld::GetPaletteOrdinal() const
{
   if (fKind != EKind::kPalettePos)
      throw std::runtime_error("This color does not represent a palette ordinal!");
   return fRedOrPalettePos;
}

bool RColorOld::AssertNotPalettePos() const
{
   if (fKind == EKind::kPalettePos) {
      throw std::runtime_error("This color does not represent a palette ordinal!");
      return false;
   }
   return true;
}

namespace {
   static RColorOld ParseRGBToColor(const std::string &strval, const std::string &name)
   {
      auto rgbalen = strval.length() - 1;
      if (rgbalen != 3 && rgbalen != 4 && rgbalen != 6 && rgbalen != 8) {
         R__ERROR_HERE("Graf2d") << "Invalid value for RColorOld default style " << name
            << " with value \"" << strval
            << "\": expect '#' followed by 3, 4, 6 or 8 hex digits (#rgb, #rgba, #rrggbbaa or #rrggbb).";
         return RColorOld::kBlack;
      }
      std::size_t pos;
      long long rgbaLL = std::stoll(strval.substr(1), &pos, /*base*/ 16);
      if (pos != 3 && pos != 4 && pos != 6 && pos != 8) {
         R__ERROR_HERE("Graf2d") << "Invalid value while parsing default style value for RColorOld " << name
            << " with value \"" << strval
            << "\": expect '#' followed by 3, 4, 6 or 8 hex digits (#rgb, #rgba, #rrggbbaa or #rrggbb).";
         return RColorOld::kBlack;
      }
      if (pos != rgbalen) {
         R__WARNING_HERE("Graf2d") << "Leftover characters while parsing default style value for RColorOld " << name
            << " with value \"" << strval << "\", remainder: \"" << strval.substr(pos - 1) << "\"";
         return RColorOld::kBlack;
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
      return RColorOld(rgba);
   }

   RColorOld ParseColorNameToColor(const std::string &strval, const std::string &name)
   {
      std::string nameLow = strval;
      std::transform(nameLow.begin(), nameLow.end(), nameLow.begin(),
         // tolower has undef behavior for char argument; cast it.
         // And must not take &stdlib function, so lambda it is.
         [](char c) { return std::tolower(static_cast<unsigned char>(c)); });
      using namespace std::string_literals;
      static const std::unordered_map<std::string, RColorOld> mapNamesToColor {
         {"red"s, RColorOld{RColorOld::kRed}},
         {"green"s, RColorOld{RColorOld::kGreen}},
         {"blue"s, RColorOld{RColorOld::kBlue}},
         {"white"s, RColorOld{RColorOld::kWhite}},
         {"black"s, RColorOld{RColorOld::kBlack}},
         {"invisible"s, RColorOld{RColorOld::kInvisible}},
         {"auto"s, RColorOld{RColorOld::kAuto}}
      };
      auto itMap = mapNamesToColor.find(nameLow);
      if (itMap == mapNamesToColor.end()) {
         R__WARNING_HERE("Graf2d") << "Cannot parse RColorOld " << name
            << " with value \"" << strval << "\": unknown color name.";
         return RColorOld::kBlack;
      }
      return itMap->second;
   }
} // unnamed namespace


constexpr RColor::RGB_t RColor::kRed;
constexpr RColor::RGB_t RColor::kGreen;
constexpr RColor::RGB_t RColor::kBlue;
constexpr RColor::RGB_t RColor::kWhite;
constexpr RColor::RGB_t RColor::kBlack;
