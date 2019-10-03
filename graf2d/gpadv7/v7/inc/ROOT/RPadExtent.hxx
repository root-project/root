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

/** \class RPadExtent
\ingroup GpadROOT7
\brief An extent / size (horizontal and vertical) in a `RPad`.
\author Axel Naumann <axel@cern.ch>
\date 2017-07-07
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/
class RPadExtent  {

   RPadLength fHoriz;   ///<  horizontal part

   RPadLength fVert;    ///<   vertical part

public:

   RPadExtent() = default;

   RPadExtent(const RPadLength& horiz, const RPadLength& vert) : RPadExtent()
   {
      fHoriz = horiz;
      fVert = vert;
   }

   RPadLength &Horiz() { return fHoriz; }
   const RPadLength &Horiz() const { return fHoriz; }

   RPadLength &Vert() { return fVert; }
   const RPadLength &Vert() const { return fVert; }


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

   /** \struct ScaleFactor
       \ingroup GpadROOT7
       \brief A scale factor (separate factors for horizontal and vertical) for scaling a `RPadLength`.
   */
   struct ScaleFactor {
      double fHoriz; ///< Horizontal scale factor
      double fVert;  ///< Vertical scale factor
   };

   /// Scale a horizontally and vertically.
   /// \param scale - the scale factor,
   RPadExtent &operator*=(const ScaleFactor &scale)
   {
      fHoriz *= scale.fHoriz;
      fVert *= scale.fVert;
      return *this;
   };
};


} // namespace Experimental
} // namespace ROOT

#endif
