/// \file ROOT/TPadPos.hxx
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

#ifndef ROOT7_TPadPos
#define ROOT7_TPadPos

#include "ROOT/TPadExtent.hxx"

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::TPadPos
  A position (horizontal and vertical) in a `TPad`.
  */
struct TPadPos: Internal::TPadHorizVert {
   using Internal::TPadHorizVert::TPadHorizVert;
   TPadPos() = default;
   TPadPos(const TPadExtent& extent): Internal::TPadHorizVert(extent) {}

   /// Add a `TPadExtent`.
   friend TPadPos operator+(const TPadPos &lhs, const TPadExtent &rhs)
   {
      return TPadPos{lhs.fHoriz + rhs.fHoriz, lhs.fVert + rhs.fVert};
   }

   /// Add to a `TPadExtent`.
   friend TPadPos operator+(const TPadExtent &lhs, const TPadPos &rhs)
   {
      return TPadPos{lhs.fHoriz + rhs.fHoriz, lhs.fVert + rhs.fVert};
   }

   /// Subtract a `TPadExtent`.
   friend TPadPos operator-(const TPadPos &lhs, const TPadExtent &rhs)
   {
      return TPadPos{lhs.fHoriz - rhs.fHoriz, lhs.fVert - rhs.fVert};
   }

   /// Subtract from a `TPadPos`s. Is this really needed?
   /*
   friend TPadPos operator-(const TPadExtent &rhs, const TPadPos &lhs)
   {
      return TPadPos{lhs.fHoriz - rhs.fHoriz, lhs.fVert - rhs.fVert};
   }
   */

   /// Add a `TPadExtent`.
   TPadPos &operator+=(const TPadExtent &rhs)
   {
      fHoriz += rhs.fHoriz;
      fVert += rhs.fVert;
      return *this;
   };

   /// Subtract a `TPadExtent`.
   TPadPos &operator-=(const TPadExtent &rhs)
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
   TPadPos &operator*=(const ScaleFactor &scale)
   {
      fHoriz *= scale.fHoriz;
      fVert *= scale.fVert;
      return *this;
   };
};

} // namespace Experimental
} // namespace ROOT

#endif
