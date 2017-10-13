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

   enum class EKind {
      kRGBA, ///< The color is defined as specific RGBA values.
      kPalettePos, ///< The color is defined as a value in the `TFrame`'s `TPalette`.
      kAuto ///< The color will be set upon drawing the canvas choosing a `TPalatte` color, see `TColor(Auto_t)`
   };

private:
   // TODO: use a `variant` here!
   /// The "R" in RGBA (0 <= R <= 1), or the palette pos if fKind is `kPalettePos`.
   float fRedOrPalettePos = 0.;

   /// The "G" in RGBA (0 <= G <= 1). Unused if `fKind != kRGBA`.
   float fGreen = 0.;

   /// The "B" in RGBA (0 <= B <= 1). Unused if `fKind != kRGBA`.
   float fBlue = 0.;

   /// The "A" in RGBA (0 <= A <= 1). Unused if `fKind != kRGBA`. `fAlpha == 0` means so transparent it's invisible,
   /// `fAlpha == 1` means completely opaque.
   float fAlpha = 1.;

   /// How the color is defined.
   EKind fKind = EKind::kRGBA;

   /// throw an exception if the color isn't specified as `kRGBA` or `kAuto`, the two cases where
   /// asking for RBGA members makes sense.
   bool AssertNotPalettePos() const;

public:
   using PredefinedRGB = std::array<float, 3>;

   // Default constructor: good old solid black.
   constexpr TColor() = default;

   /// Initialize a TColor with red, green, blue and alpha component.
   constexpr TColor(float r, float g, float b, float alpha): fRedOrPalettePos(r), fGreen(g), fBlue(b), fAlpha(alpha) {}

   /// Initialize a TColor with red, green, blue and alpha component.
   constexpr TColor(float r, float g, float b, Alpha alpha = kOpaque): TColor(r, g, b, alpha.fVal) {}

   /// Initialize a TColor with red, green, blue and alpha component.
   constexpr TColor(const PredefinedRGB &predef): TColor(predef[0], predef[1], predef[2]) {}

   /// Initialize a `TColor` with a `TPalette` ordinal. The actual color is determined from the pad's
   /// (or rather its `TFrame`'s) `TPalette`
   constexpr TColor(float paletteOrdinal): fRedOrPalettePos(paletteOrdinal), fKind(EKind::kPalettePos) {}

   /**\class AutoTag
    Used to signal that this color shall be automatically chosen by the drawing routines, by picking a color
    from the `TPad`'s (or rather its `TFrame`'s) current `TPalette`.
   */
   class AutoTag {};

   /// Constructs an automatically assigned color. Call as `TColor col(TColor::kAuto)`.
   constexpr TColor(AutoTag): fKind(EKind::kAuto) {}

   /// Determine whether this TColor is storing RGBA (in contrast to an ordinal of a TPalette).
   bool IsRGBA() const { return fKind == EKind::kRGBA; }

   /// Determine whether this `TColor` is storing an ordinal of a TPalette (in contrast to RGBA).
   bool IsPaletteOrdinal() const { return fKind == EKind::kPalettePos; }

   /// Determine whether this `TColor` will be assigned a actual color upon drawing.
   bool IsAuto() const { return fKind == EKind::kAuto; }

   /// If this is an ordinal in a palette, resolve the
   float GetPaletteOrdinal() const;

   friend bool operator==(const TColor &lhs, const TColor &rhs)
   {
      if (lhs.fKind != rhs.fKind)
         return false;
      switch (lhs.fKind) {
      case EKind::kPalettePos:
         return lhs.fRedOrPalettePos == rhs.fRedOrPalettePos;
      case EKind::kRGBA:
         return lhs.fRedOrPalettePos == rhs.fRedOrPalettePos && lhs.fGreen == rhs.fGreen && lhs.fBlue == rhs.fBlue &&
             lhs.fAlpha == rhs.fAlpha;
      case EKind::kAuto:
         return true; // is that what we need?
      }
      return false;
   }

   /// For RGBA or auto colors, get the red component (0..1).
   float GetRed() const {
      if (AssertNotPalettePos())
         return fRedOrPalettePos;
      return 0.;
   }

   /// For RGBA or auto colors, get the green component (0..1).
   float GetGreen() const {
      if (AssertNotPalettePos())
         return fGreen;
      return 0.;
   }

   /// For RGBA or auto colors, get the blue component (0..1).
   float GetBlue() const {
      if (AssertNotPalettePos())
         return fBlue;
      return 0.;
   }

   /// For RGBA or auto colors, get the alpha component (0..1).
   float GetAlpha() const {
      if (AssertNotPalettePos())
         return fAlpha;
      return 0.;
   }

   /// For RGBA or auto colors, set the red component.
   void SetRed(float r) {
      if (AssertNotPalettePos())
         fRedOrPalettePos = r;
   }

   /// For RGBA or auto colors, set the green component.
   void SetGreen(float g) {
      if (AssertNotPalettePos())
         fGreen = g;
   }

   /// For RGBA or auto colors, set the blue component.
   void SetBlue(float b) {
      if (AssertNotPalettePos())
         fBlue = b;
   }

   /// For RGBA or auto colors, set the alpha component.
   void SetAlpha(float a) {
      if (AssertNotPalettePos())
         fAlpha = a;
   }

   /// For RGBA or auto colors, set the alpha component.
   void SetAlpha(Alpha a) {
      if (AssertNotPalettePos())
         fAlpha = (float)a;
   }

   ///\{
   ///\name Default colors
   static constexpr PredefinedRGB kRed{{1., 0., 0.}};
   static constexpr PredefinedRGB kGreen{{0., 1., 0.}};
   static constexpr PredefinedRGB kBlue{{0., 0, 1.}};
   static constexpr PredefinedRGB kWhite{{1., 1, 1.}};
   static constexpr PredefinedRGB kBlack{{0., 0., 0.}};
   static constexpr AutoTag kAuto{};
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
