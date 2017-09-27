/// \file ROOT/TPalette.hxx
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

#ifndef ROOT7_TPalette
#define ROOT7_TPalette

#include <RStringView.h>
#include <ROOT/TColor.hxx>

#include <utility>
#include <vector>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::TPalette
  A set of colors. `TColor`s can be conveniently generated from this.

  A palette associates a color with an ordinal number: for a normalized palette,
  this number ranges from 0..1. For user-valued palettes, the palette yields a color for
  user-coordinates (for instance histogram content), in an arbitrary range.
  */
class TPalette {
   /// Palette colors: the color points and their ordinal value.
   std::vector<std::pair<double, TColor>> fColors;

   /// Whether to interpolate between the colors, or to pick one of fColors.
   bool fInterpolate = true;

   /// Whether the palette's ordinal numbers are normalized.
   bool fNormalized = true;

public:
   /// Tag type used to signal that the palette's colors should not be interpolated.
   struct Discrete_t {
   };

   /// Tag value used to signal that the palette's colors should not be interpolated. Can be passed to the
   /// constructor: `TPalette palette(TPalette::Discrete, {{-100., TColor::kWhite}, {100., TColor::kRed}})`
   static constexpr const Discrete_t Discrete{};

   /// Construct a TPalette from a vector of (ordinal|color) pairs as interpolation points.
   /// Palette colors will be these points for the ordinal passed in as `first` of the `pair`,
   /// and interpolated in between the ordinal points. The points will be sorted.
   /// The palette is normalized if the lowest ordinal is 0. and the highest ordinal is 1.;
   /// otherwise, the palette is a user-valued palette.
   TPalette(const std::vector<std::pair<double, TColor>> &interpPoints);

   /// Construct a TPalette from a vector of (ordinal|color) pairs. For a given value, the palette returns
   /// the color with an ordinal that is closest to the value. The points will be sorted.
   /// The palette is normalized if the lowest ordinal is 0. and the highest ordinal is 1.;
   /// otherwise, the palette is a user-valued palette.
   TPalette(Discrete_t, const std::vector<std::pair<double, TColor>> &interpPoints);

   /// Construct a normalized TPalette from a vector of colors as interpolation points. The ordinal associated
   /// with each color is equidistant from 0..1, i.e. for three colors it will be 0., 0.5 and 1, respectively.
   /// Palette colors will be these points for the ordinal associated with the color,
   /// and interpolated in between the ordinal points.
   TPalette(const std::vector<TColor> &interpPoints);

   /// Construct a normalized TPalette from a vector of colors. The ordinal associated
   /// with each color is equidistant from 0..1, i.e. for three colors it will be 0., 0.5 and 1, respectively.
   /// For a given value, the palette returns the color with an ordinal that is closest to the value.
   TPalette(Discrete_t, const std::vector<TColor> &interpPoints);
};

} // namespace Experimental
} // namespace ROOT

#endif
