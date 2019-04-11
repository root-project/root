/// \file ROOT/RPadPos.hxx
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

#ifndef ROOT7_RPadPos
#define ROOT7_RPadPos

#include "ROOT/RPadExtent.hxx"

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::RPadPos
  A position (horizontal and vertical) in a `RPad`.
  */
struct RPadPos: Internal::RPadHorizVert {
   using Internal::RPadHorizVert::RPadHorizVert;
   RPadPos() = default;
   RPadPos(const RPadExtent& extent): Internal::RPadHorizVert(extent) {}

   /// Add a `RPadExtent`.
   friend RPadPos operator+(const RPadPos &lhs, const RPadExtent &rhs)
   {
      return RPadPos{lhs.fHoriz + rhs.fHoriz, lhs.fVert + rhs.fVert};
   }

   /// Add to a `RPadExtent`.
   friend RPadPos operator+(const RPadExtent &lhs, const RPadPos &rhs)
   {
      return RPadPos{lhs.fHoriz + rhs.fHoriz, lhs.fVert + rhs.fVert};
   }

   /// Subtract a `RPadExtent`.
   friend RPadPos operator-(const RPadPos &lhs, const RPadExtent &rhs)
   {
      return RPadPos{lhs.fHoriz - rhs.fHoriz, lhs.fVert - rhs.fVert};
   }

   /// Subtract from a `RPadPos`s. Is this really needed?
   /*
   friend RPadPos operator-(const RPadExtent &rhs, const RPadPos &lhs)
   {
      return RPadPos{lhs.fHoriz - rhs.fHoriz, lhs.fVert - rhs.fVert};
   }
   */

   /// Add a `RPadExtent`.
   RPadPos &operator+=(const RPadExtent &rhs)
   {
      fHoriz += rhs.fHoriz;
      fVert += rhs.fVert;
      return *this;
   };

   /// Subtract a `RPadExtent`.
   RPadPos &operator-=(const RPadExtent &rhs)
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

   /// Scale a `RPadHorizVert` horizontally and vertically.
   /// \param scale - the scale factor,
   RPadPos &operator*=(const ScaleFactor &scale)
   {
      fHoriz *= scale.fHoriz;
      fVert *= scale.fVert;
      return *this;
   };
};

/// Initialize a RPadPos from a style string.
RPadPos FromAttributeString(const std::string &val, const std::string &name, RPadPos*);
/// Convert a RPadPos to a std::string, suitable for PosFromString().
std::string ToAttributeString(const RPadPos &pos);

} // namespace Experimental
} // namespace ROOT

#endif
