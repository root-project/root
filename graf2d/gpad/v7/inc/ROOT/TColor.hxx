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

#ifndef ROOT7_TColor
#define ROOT7_TColor

#include <array>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::TColor
  A color: Red|Green|Blue|Alpha, or a position in a TPalette
  */
class TColor {
public:
   /** \class ROOT::Experimental::TColor::TAlpha
    The alpha value of a color: 0 is completely transparent, 1 is completely opaque.
    */
   struct Alpha {
      float fVal;
      explicit operator float() const { return fVal; }
   };
   /// An opaque color.
   static constexpr Alpha kOpaque{1.};
   /// A completely transparent color.
   static constexpr Alpha kTransparent{0.};

private:
   /// The "R" in RGBA (0 <= R <= 1), or the palette pos if fIsRGBA is `false`.
   float fRedOrPalettePos = 0.;

   /// The "G" in RGBA (0 <= G <= 1). Unused if `!fIsRGBA`.
   float fGreen = 0.;

   /// The "B" in RGBA (0 <= B <= 1). Unused if `!fIsRGBA`.
   float fBlue = 0.;

   /// The "A" in RGBA (0 <= A <= 1). Unused if `!fIsRGBA`. `fAlpha == 0` means so transparent it's invisible,
   /// `fAlpha == 1` means completely opaque.
   float fAlpha = 1.;

   /// Whether this is an RGBA color or an index in the `TPad`'s `TPalette`.
   bool fIsRGBA = true;

public:
   using Predefined = std::array<float, 3>;

   // Default constructor: good old solid black.
   constexpr TColor() = default;

   /// Initialize a TColor with red, green, blue and alpha component.
   constexpr TColor(float r, float g, float b, float alpha): fRedOrPalettePos(r), fGreen(g), fBlue(b), fAlpha(alpha) {}

   /// Initialize a TColor with red, green, blue and alpha component.
   constexpr TColor(float r, float g, float b, Alpha alpha = kOpaque): TColor(r, g, b, alpha.fVal) {}

   /// Initialize a TColor with red, green, blue and alpha component.
   constexpr TColor(const Predefined &predef): TColor(predef[0], predef[1], predef[2]) {}

   friend bool operator==(const TColor &lhs, const TColor &rhs)
   {
      if (lhs.fIsRGBA != rhs.fIsRGBA)
         return false;
      if (lhs.fIsRGBA)
         return lhs.fRedOrPalettePos == rhs.fRedOrPalettePos;
      return lhs.fRedOrPalettePos == rhs.fRedOrPalettePos && lhs.fGreen == rhs.fGreen && lhs.fBlue == rhs.fBlue &&
             lhs.fAlpha == rhs.fAlpha;
   }

   ///\{
   ///\name Default colors

   // Implemented in TPalette.cxx.
   static constexpr Predefined kRed{{0.5, 0., 0.}};
   static constexpr Predefined kGreen{{0., 0.5, 0.}};
   static constexpr Predefined kBlue{{0., 0, 0.5}};
   static constexpr Predefined kWhite{{1., 1, 1.}};
   static constexpr Predefined kBlack{{0., 0., 0.}};
   ///\}
};

#if 0
// TODO: see also imagemagick's C++ interface for TColor operations!
// https://www.imagemagick.org/api/magick++-classes.php

/// User-defined literal to add alpha to a `TColor`
///
/// Use as
/// ```
/// using namespace ROOT::Experimental;
/// TColor red = TColor::kRed + 0.2_alpha;
/// ```
inline TPadCoord::Normal operator"" _alpha(long double val)
{
   return TPadCoord::Normal{(double)val};
}
#endif

} // namespace Experimental
} // namespace ROOT

#endif
