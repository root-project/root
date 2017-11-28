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

  A palette can be a smooth gradients by interpolation of support points, or a set of
  discrete colors.
  */
class TPalette {
public:
   /** \class ROOT::Experimental::TPalette::OrdinalAndColor
    An ordinal value and its associated color.
    */
   struct OrdinalAndColor {
      double fOrdinal; ///< The value associated with the color.
      TColor fColor;   ///< The color associated with the value.

      /// Compare two `OrdinalAndColor`s, for sorting.
      friend bool operator<(const OrdinalAndColor &lhs, const OrdinalAndColor &rhs)
      {
         return lhs.fOrdinal < rhs.fOrdinal;
      }
      /// Compare an `OrdinalAndColor` and an ordinal value.
      friend bool operator<(const OrdinalAndColor &lhs, double rhs) { return lhs.fOrdinal < rhs; }
   };

private:
   /// Palette colors: the color points and their ordinal value.
   std::vector<OrdinalAndColor> fColors;

   /// Whether to interpolate between the colors (in contrast to picking one of fColors).
   bool fInterpolate = true;

   /// Whether the palette's ordinal numbers are normalized.
   bool fNormalized = true;

   TPalette(bool interpolate, bool knownNormalized, const std::vector<OrdinalAndColor> &points);
   TPalette(bool interpolate, const std::vector<TColor> &points);

public:
   /// Tag type used to signal that the palette's colors should not be interpolated.
   struct Discrete_t {
   };

   /// Tag value used to signal that the palette's colors should not be interpolated. Can be passed to the
   /// constructor: `TPalette palette(TPalette::kDiscrete, {{-100., TColor::kWhite}, {100., TColor::kRed}})`
   static constexpr const Discrete_t kDiscrete{};

   TPalette() = default;

   /// Construct a TPalette from a vector of (ordinal|color) pairs as interpolation points.
   /// Palette colors will be these points for the ordinal, and interpolated in between the
   /// ordinal points. The points will be sorted.
   /// The palette is normalized if the lowest ordinal is 0. and the highest ordinal is 1.;
   /// otherwise, the palette is a user-valued palette.
   TPalette(const std::vector<OrdinalAndColor> &interpPoints): TPalette(true, false, interpPoints) {}

   /// Construct a TPalette from a vector of (ordinal|color) pairs. For a given value, the palette returns
   /// the color with an ordinal that is closest to the value. The points will be sorted.
   /// The palette is normalized if the lowest ordinal is 0. and the highest ordinal is 1.;
   /// otherwise, the palette is a user-valued palette.
   TPalette(Discrete_t, const std::vector<OrdinalAndColor> &points): TPalette(false, false, points) {}

   /// Construct a normalized TPalette from a vector of colors as interpolation points. The ordinal associated
   /// with each color is equidistant from 0..1, i.e. for three colors it will be 0., 0.5 and 1, respectively.
   /// Palette colors will be these points for the ordinal associated with the color,
   /// and interpolated in between the ordinal points.
   TPalette(const std::vector<TColor> &interpPoints): TPalette(true, interpPoints) {}

   /// Construct a normalized TPalette from a vector of colors. The ordinal associated
   /// with each color is equidistant from 0..1, i.e. for three colors it will be 0., 0.5 and 1, respectively.
   /// For a given value, the palette returns the color with an ordinal that is closest to the value.
   TPalette(Discrete_t, const std::vector<TColor> &points): TPalette(false, points) {}

   /// Whether the palette is normalized, i.e. covers colors in the ordinal range 0..1.
   bool IsNormalized() const { return fNormalized; }

   /// Whether the palette is discrete, i.e. does no interpolation between colors.
   bool IsDiscrete() const { return !fInterpolate; }

   /// Whether the palette is a smooth gradient generated by interpolating between the color points.
   bool IsGradient() const { return fInterpolate; }

   /// Get the color associated with the ordinal value. The value is expected to be 0..1 for a normalized
   /// palette.
   TColor GetColor(double ordinal);

   /// Given a TColor (that might either be a RGBA or a TPalette ordinal), get the RGBA-based color.
   TColor ResolveRGBAColor(const TColor &col)
   {
      if (col.IsRGBA())
         return col;
      return GetColor(col.GetPaletteOrdinal());
   }

   ///\{
   ///\name Global Palettes

   /// Register a palette in the set of global palettes, making it available to `GetPalette()`.
   /// This function is not thread safe; any concurrent call to global Palette manipulation must be synchronized!
   static void RegisterPalette(std::string_view name, const TPalette &palette);

   /// Get a global palette by name. Returns an empty palette if no palette with that name is known.
   /// This function is not thread safe; any concurrent call to global Palette manipulation must be synchronized!
   static const TPalette &GetPalette(std::string_view name);

   ///\}
};

} // namespace Experimental
} // namespace ROOT

#endif
