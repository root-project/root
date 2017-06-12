/// \file ROOT/THistData.h
/// \ingroup Hist ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2016-06-01
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_THistUtils_h
#define ROOT7_THistUtils_h

#include <array>
#include <type_traits>

namespace ROOT {
namespace Experimental {
namespace Hist {

template <int DIMENSIONS>
struct TCoordArray: std::array<double, DIMENSIONS> {
   using Base_t = std::array<double, DIMENSIONS>;

   /// Default construction.
   TCoordArray() = default;

   /// Construction with one `double` per `DIMENSION`.
   template<class...ELEMENTS, class = typename std::enable_if<sizeof...(ELEMENTS) + 1 == DIMENSIONS>::type>
   TCoordArray(double x, ELEMENTS...el): Base_t{{x, el...}} {}

   /// Fallback constructor, invoked if the one above fails because of the wrong number of
   /// arguments / coordinates.
   template<class T, class...ELEMENTS, class = typename std::enable_if<sizeof...(ELEMENTS) + 1 != DIMENSIONS>::type>
   TCoordArray(T, ELEMENTS...) {
      static_assert(sizeof...(ELEMENTS) + 1 == DIMENSIONS, "Number of coordinates does not match DIMENSIONS");
   }

   /// Construction from a C-style array.
   TCoordArray(double (&arr)[DIMENSIONS]): Base_t(arr) {}
};

template <int DIMENSIONS>
//using CoordArray_t = std::array<double, DIMENSIONS>;
using CoordArray_t = TCoordArray<DIMENSIONS>;


} // namespace Hist
} // namespace Experimental
} // namespace ROOT

#endif //ROOT7_THistUtils_h
