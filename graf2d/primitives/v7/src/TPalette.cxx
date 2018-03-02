/// \file TPalette.cxx
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

#include "ROOT/TPalette.hxx"

#include "ROOT/TLogger.hxx"

#include <algorithm>
#include <cmath>
#include <exception>
#include <unordered_map>

using namespace ROOT::Experimental;

TPalette::TPalette(bool interpolate, bool knownNormalized, const std::vector<TPalette::OrdinalAndColor> &points)
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
static std::vector<TPalette::OrdinalAndColor> AddOrdinals(const std::vector<TColor> &points)
{
   std::vector<TPalette::OrdinalAndColor> ret(points.size());
   auto addOneOrdinal = [&](const TColor &col) -> TPalette::OrdinalAndColor {
      return {1. / (points.size() - 1) * (&col - points.data()), col};
   };
   std::transform(points.begin(), points.end(), ret.begin(), addOneOrdinal);
   return ret;
}
} // unnamed namespace

TPalette::TPalette(bool interpolate, const std::vector<TColor> &points)
   : TPalette(interpolate, true, AddOrdinals(points))
{}

TColor TPalette::GetColor(double ordinal)
{
   if (fInterpolate) {
      R__ERROR_HERE("Gpad") << "Not yet implemented!";
   } else {
      auto iColor = std::lower_bound(fColors.begin(), fColors.end(), ordinal);
      if (iColor == fColors.end())
         return fColors.back().fColor;
      // Is iColor-1 closer to ordinal than iColor?
      if ((iColor - 1)->fOrdinal - ordinal < ordinal - iColor->fOrdinal)
         return (iColor - 1)->fColor;
      return iColor->fColor;
   }
   return TColor{};
}

namespace {
using GlobalPalettes_t = std::unordered_map<std::string, TPalette>;
static GlobalPalettes_t CreateDefaultPalettes()
{
   GlobalPalettes_t ret;
   ret["default"] = TPalette({TColor::kRed, TColor::kBlue});
   ret["bw"] = TPalette({TColor::kBlack, TColor::kWhite});
   return ret;
}

static GlobalPalettes_t &GetGlobalPalettes()
{
   static GlobalPalettes_t globalPalettes = CreateDefaultPalettes();
   return globalPalettes;
}
} // unnamed namespace

void TPalette::RegisterPalette(std::string_view name, const TPalette &palette)
{
   GetGlobalPalettes()[std::string(name)] = palette;
}

const TPalette &TPalette::GetPalette(std::string_view name)
{
   static const TPalette sNoPaletteWithThatName;
   auto iGlobalPalette = GetGlobalPalettes().find(std::string(name));
   if (iGlobalPalette == GetGlobalPalettes().end())
      return sNoPaletteWithThatName;
   return iGlobalPalette->second;
}
