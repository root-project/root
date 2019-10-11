/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RPalette.hxx"

#include "ROOT/RLogger.hxx"

#include <algorithm>
#include <cmath>
#include <exception>
#include <unordered_map>

using namespace ROOT::Experimental;

RPalette::RPalette(bool interpolate, bool knownNormalized, const std::vector<RPalette::OrdinalAndColor> &points)
   : fColors(points), fInterpolate(interpolate), fNormalized(knownNormalized)
{
   if (points.size() < 2)
      throw std::runtime_error("Must have at least two points to build a palette!");

   std::sort(fColors.begin(), fColors.end());

   if (!knownNormalized) {
      // Is this a normalized palette? I.e. are the first and last ordinals 0 and 1?
      double high = fColors.back().fOrdinal;
      double low = fColors.front().fOrdinal;
      double prec = (high - low) * 1E-6;

      auto reasonablyEqual = [&](double val, double expected) -> bool { return std::fabs(val - expected) < prec; };
      fNormalized = reasonablyEqual(low, 0.) && reasonablyEqual(high, 1.);
   }
}

namespace {
static std::vector<RPalette::OrdinalAndColor> AddOrdinals(const std::vector<RColor> &points)
{
   std::vector<RPalette::OrdinalAndColor> ret(points.size());
   auto addOneOrdinal = [&](const RColor &col) -> RPalette::OrdinalAndColor {
      return {1. / (points.size() - 1) * (&col - points.data()), col};
   };
   std::transform(points.begin(), points.end(), ret.begin(), addOneOrdinal);
   return ret;
}
} // unnamed namespace

RPalette::RPalette(bool interpolate, const std::vector<RColor> &points)
   : RPalette(interpolate, true, AddOrdinals(points))
{}

RColor RPalette::GetColor(double ordinal)
{
   if (fInterpolate) {
      R__ERROR_HERE("Gpad") << "Not yet implemented!";
   } else {
      auto iColor = std::lower_bound(fColors.begin(), fColors.end(), ordinal);
      if (iColor == fColors.end())
         return fColors.back().fColor;
      // Is iColor-1 closer to ordinal than iColor?
      if (iColor != fColors.begin() && (iColor - 1)->fOrdinal - ordinal < ordinal - iColor->fOrdinal)
         return (iColor - 1)->fColor;
      return iColor->fColor;
   }
   return RColor{};
}

namespace {
using GlobalPalettes_t = std::unordered_map<std::string, RPalette>;
static GlobalPalettes_t CreateDefaultPalettes()
{
   GlobalPalettes_t ret;
   ret["default"] = RPalette({RColor::kRed, RColor::kBlue});
   ret["bw"] = RPalette({RColor::kBlack, RColor::kWhite});
   return ret;
}

static GlobalPalettes_t &GetGlobalPalettes()
{
   static GlobalPalettes_t globalPalettes = CreateDefaultPalettes();
   return globalPalettes;
}
} // unnamed namespace

void RPalette::RegisterPalette(std::string_view name, const RPalette &palette)
{
   GetGlobalPalettes()[std::string(name)] = palette;
}

const RPalette &RPalette::GetPalette(std::string_view name)
{
   static const RPalette sNoPaletteWithThatName;
   auto iGlobalPalette = GetGlobalPalettes().find(std::string(name));
   if (iGlobalPalette == GetGlobalPalettes().end())
      return sNoPaletteWithThatName;
   return iGlobalPalette->second;
}
