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

#include <RStringView.h>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::TColor
  A color: Red|Green|Blue|Alpha, or an position in a TPalette
  */
class TColor {
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
   // Default constructor: good old solid black.
   TColor() = default;

   /// Initialize a TColor with red, green, blue and alpha component.
   TColor(float r, float g, float b, float alpha = 1.): fRedOrPalettePos(r), fGreen(g), fBlue(b), fAlpha(alpha) {}

   bool operator==(const TColor &rhs)
   {
      if (fIsRGBA != rhs.fIsRGBA)
         return false;
      if (fIsRGBA)
         return fRedOrPalettePos == rhs.fRedOrPalettePos;
      return fRedOrPalettePos == rhs.fRedOrPalettePos && fGreen == rhs.fGreen && fBlue == rhs.fBlue &&
             fAlpha == rhs.fAlpha;
   }
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
