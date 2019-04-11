/// \file ROOT/RPadExtent.hxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2017-07-07
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RPadExtent
#define ROOT7_RPadExtent

#include "ROOT/RPadLength.hxx"

#include <array>
#include <string>

namespace ROOT {
namespace Experimental {

namespace Internal {
/** \class ROOT::Experimental::Internal::RPadHorizVert
   A 2D (horizontal and vertical) combination of `RPadLength`s.
   */

struct RPadHorizVert {
   RPadLength fHoriz; ///< Horizontal position
   RPadLength fVert;  ///< Vertical position

   RPadHorizVert() = default;
   RPadHorizVert(const std::array<RPadLength, 2> &hv): fHoriz(hv[0]), fVert(hv[1]) {}
   RPadHorizVert(const RPadLength &horiz, const RPadLength &vert): fHoriz(horiz), fVert(vert) {}

   void SetFromAttrString(const std::string &val, const std::string &name);
};
}; // namespace Internal

/** \class ROOT::Experimental::RPadExtent
  An extent / size (horizontal and vertical) in a `RPad`.
  */
struct RPadExtent: Internal::RPadHorizVert {
   using Internal::RPadHorizVert::RPadHorizVert;

   /// Add two `RPadExtent`s.
   friend RPadExtent operator+(RPadExtent lhs, const RPadExtent &rhs)
   {
      return {lhs.fHoriz + rhs.fHoriz, lhs.fVert + rhs.fVert};
   }

   /// Subtract two `RPadExtent`s.
   friend RPadExtent operator-(RPadExtent lhs, const RPadExtent &rhs)
   {
      return {lhs.fHoriz - rhs.fHoriz, lhs.fVert - rhs.fVert};
   }

   /// Add a `RPadExtent`.
   RPadExtent &operator+=(const RPadExtent &rhs)
   {
      fHoriz += rhs.fHoriz;
      fVert += rhs.fVert;
      return *this;
   };

   /// Subtract a `RPadExtent`.
   RPadExtent &operator-=(const RPadExtent &rhs)
   {
      fHoriz -= rhs.fHoriz;
      fVert -= rhs.fVert;
      return *this;
   };

   /** \class ScaleFactor
      A scale factor (separate factors for horizontal and vertical) for scaling a `RPadLength`.
      */
   struct ScaleFactor {
      double fHoriz; ///< Horizontal scale factor
      double fVert;  ///< Vertical scale factor
   };

   /// Scale a `RPadHorizVert` horizonally and vertically.
   /// \param scale - the scale factor,
   RPadExtent &operator*=(const ScaleFactor &scale)
   {
      fHoriz *= scale.fHoriz;
      fVert *= scale.fVert;
      return *this;
   };
};

/// Initialize a RPadExtent from a style string.
/// Syntax: X, Y
/// where X and Y are a series of numbers separated by "+", where each number is followed by one of
/// `px`, `user`, `normal` to specify an extent in pixel, user or normal coordinates. Spaces between
/// any part is allowed.
/// Example: `100 px + 0.1 user, 0.5 normal` is a `RPadExtent{100_px + 0.1_user, 0.5_normal}`.

RPadExtent FromAttributeString(const std::string &val, const std::string &name, RPadExtent*);

std::string ToAttributeString(const RPadExtent &extent);

} // namespace Experimental
} // namespace ROOT

#endif
