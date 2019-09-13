/// \file ROOT/RMarker.hxx
/// \ingroup Graf ROOT7
/// \author Olivier Couet <Olivier.Couet@cern.ch>
/// \date 2017-10-16
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RMarker
#define ROOT7_RMarker

#include <ROOT/RDrawable.hxx>
#include <ROOT/RAttrMarker.hxx>
#include <ROOT/RPadPos.hxx>

#include <initializer_list>
#include <memory>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::RMarker
 A simple marker.
 */

class RMarker : public RDrawable {

   RDrawableAttributes fAttr{"marker"};         ///< attributes
   RPadPos fP{fAttr, "pos_"};                  ///<! position
   RAttrMarker fMarkerAttr{fAttr, "marker_"};  ///<! marker attributes

public:

   RMarker() = default;

   RMarker(const RPadPos& p) : RMarker() { fP = p; }

   void SetP(const RPadPos& p) { fP = p; }
   const RPadPos& GetP() const { return fP; }

   RAttrMarker &AttrMarker() { return fMarkerAttr; }
   const RAttrMarker &AttrMarker() const { return fMarkerAttr; }

};

} // namespace Experimental
} // namespace ROOT

#endif
