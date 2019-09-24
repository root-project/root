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

#include <array>
#include <string>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::RPadPos
  A position (horizontal and vertical) in a `RPad`.
  */
class RPadPos {

   RPadLength fHoriz;   ///<  horizontal part

   RPadLength fVert;    ///<   vertical part

public:

   RPadPos() = default;

   RPadPos(const RPadLength& horiz, const RPadLength& vert) : RPadPos()
   {
      fHoriz = horiz;
      fVert = vert;
   }

   RPadPos(const RPadExtent &rhs) : RPadPos()
   {
      fHoriz = rhs.Horiz();
      fVert = rhs.Vert();
   }

   RPadLength &Horiz() { return fHoriz; }
   const RPadLength &Horiz() const { return fHoriz; }

   RPadLength &Vert() { return fVert; }
   const RPadLength &Vert() const { return fVert; }


   /// Add two `RPadPos`s.
   RPadPos &operator=(const RPadExtent &rhs)
   {
      fHoriz = rhs.Horiz();
      fVert = rhs.Vert();
      return *this;
   }


   /// Add two `RPadPos`s.
   friend RPadPos operator+(RPadPos lhs, const RPadExtent &rhs)
   {
      return {lhs.fHoriz + rhs.Horiz(), lhs.fVert + rhs.Vert()};
   }

   /// Subtract two `RPadPos`s.
   friend RPadPos operator-(RPadPos lhs, const RPadExtent &rhs)
   {
      return {lhs.fHoriz - rhs.Horiz(), lhs.fVert - rhs.Vert()};
   }

   /// Add a `RPadPos`.
   RPadPos &operator+=(const RPadExtent &rhs)
   {
      fHoriz += rhs.Horiz();
      fVert += rhs.Vert();
      return *this;
   };

   /// Subtract a `RPadPos`.
   RPadPos &operator-=(const RPadExtent &rhs)
   {
      fHoriz -= rhs.Horiz();
      fVert -= rhs.Vert();
      return *this;
   };

   /** \class ScaleFactor
      A scale factor (separate factors for horizontal and vertical) for scaling a `RPadLength`.
      */
   struct ScaleFactor {
      double fHoriz; ///< Horizontal scale factor
      double fVert;  ///< Vertical scale factor
   };

   /// Scale a horizontally and vertically.
   /// \param scale - the scale factor,
   RPadPos &operator*=(const ScaleFactor &scale)
   {
      fHoriz *= scale.fHoriz;
      fVert *= scale.fVert;
      return *this;
   };
};

}
}

#endif
