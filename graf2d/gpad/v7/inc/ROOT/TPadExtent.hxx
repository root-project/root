/// \file ROOT/TPadExtent.hxx
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

#ifndef ROOT7_TPadExtent
#define ROOT7_TPadExtent

#include "ROOT/TPadCoord.hxx"

#include <array>

namespace ROOT {
namespace Experimental {

namespace Internal {
/** \class ROOT::Experimental::Internal::TPadHorizVert
   A 2D (horizontal and vertical) combination of `TPadCoord`s.
   */

struct TPadHorizVert {
   TPadCoord fHoriz; ///< Horizontal position
   TPadCoord fVert;  ///< Vertical position

   TPadHorizVert() = default;
   TPadHorizVert(const std::array<TPadCoord, 2> &hv): fHoriz(hv[0]), fVert(hv[1]) {}
   TPadHorizVert(const TPadCoord &horiz, const TPadCoord &vert): fHoriz(horiz), fVert(vert) {}
};
}; // namespace Internal

/** \class ROOT::Experimental::TPadExtent
  An extent / size (horizontal and vertical) in a `TPad`.
  */
struct TPadExtent: Internal::TPadHorizVert {
   using Internal::TPadHorizVert::TPadHorizVert;

   /// Add two `TPadPos`s.
   friend TPadExtent operator+(TPadExtent lhs, const TPadExtent &rhs)
   {
      return {lhs.fHoriz + rhs.fHoriz, lhs.fVert + rhs.fVert};
   }

   /// Subtract two `TPadPos`s.
   friend TPadExtent operator-(TPadExtent lhs, const TPadExtent &rhs)
   {
      return {lhs.fHoriz - rhs.fHoriz, lhs.fVert - rhs.fVert};
   }

   /// Add a `TPadPos`.
   TPadExtent &operator+=(const TPadExtent &rhs)
   {
      fHoriz += rhs.fHoriz;
      fVert += rhs.fVert;
      return *this;
   };

   /// Subtract a `TPadPos`.
   TPadExtent &operator-=(const TPadExtent &rhs)
   {
      fHoriz -= rhs.fHoriz;
      fVert -= rhs.fVert;
      return *this;
   };

   /** \class ScaleFactor
      A scale factor (separate factors for horizontal and vertical) for scaling a `TPadCoord`.
      */
   struct ScaleFactor {
      double fHoriz; ///< Horizontal scale factor
      double fVert;  ///< Vertical scale factor
   };

   /// Scale a `TPadHorizVert` horizonally and vertically.
   /// \param scale - the scale factor,
   TPadExtent &operator*=(const ScaleFactor &scale)
   {
      fHoriz *= scale.fHoriz;
      fVert *= scale.fVert;
      return *this;
   };
};

} // namespace Experimental
} // namespace ROOT

#endif
