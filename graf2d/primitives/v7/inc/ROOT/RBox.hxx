/// \file ROOT/RBox.hxx
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

#ifndef ROOT7_RBox
#define ROOT7_RBox

#include <ROOT/RDrawable.hxx>
#include <ROOT/RAttrBox.hxx>
#include <ROOT/RPadPos.hxx>

#include <initializer_list>
#include <memory>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::RBox
 A simple box.
 */

class RBox : public RDrawable {

   /// Box's coordinates
   RPadPos fP1;                         ///< line begin
   RPadPos fP2;                         ///< line end
   RAttrValues fAttr{"line"};           ///< attributes
   RAttrBox fBoxAttr{fAttr, "box_"};    ///<! line attributes

public:

   RBox() = default;

   RBox(const RPadPos& p1, const RPadPos& p2) : RBox() { fP1 = p1; fP2 = p2; }

   RBox &SetP1(const RPadPos& p1) { fP1 = p1; return *this; }
   RBox &SetP2(const RPadPos& p2) { fP2 = p2; return *this; }

   const RPadPos& GetP1() const { return fP1; }
   const RPadPos& GetP2() const { return fP2; }

   RAttrBox &AttrBox() { return fBoxAttr; }
   const RAttrBox &AttrBox() const { return fBoxAttr; }
};

} // namespace Experimental
} // namespace ROOT

#endif
