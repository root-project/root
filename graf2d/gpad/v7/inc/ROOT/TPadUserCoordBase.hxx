/// \file ROOT/TPadUserCoord.hxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2017-07-15
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_TPadUserCoordBase
#define ROOT7_TPadUserCoordBase

#include <ROOT/TPadCoord.hxx>

#include <array>

namespace ROOT {
namespace Experimental {

namespace Detail {

/** \class ROOT::Experimental::Internal::Detail::TPadUserCoordBase
  Base class for user coordinates (e.g. for histograms) used by `TPad` and `TCanvas`.
  */

class TPadUserCoordBase {
private:
   /// Disable copy construction.
   TPadUserCoordBase(const TPadUserCoordBase &) = delete;

   /// Disable assignment.
   TPadUserCoordBase &operator=(const TPadUserCoordBase &) = delete;

protected:
   /// Allow derived classes to default construct a TPadUserCoordBase.
   TPadUserCoordBase() = default;

public:
   virtual ~TPadUserCoordBase();

   /// Convert user coordinates to normal coordinates.
   virtual std::array<TPadCoord::Normal, 2> ToNormal(const std::array<TPadCoord::User, 2> &) const = 0;
};

} // namespace Detail
} // namespace Experimental
} // namespace ROOT

#endif
