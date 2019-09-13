/// \file ROOT/RAttrBox.hxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2018-10-17
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RAttrBox
#define ROOT7_RAttrBox

#include <ROOT/RDrawingAttr.hxx>
#include <ROOT/RAttrLine.hxx>
#include <ROOT/RColor.hxx>
#include <ROOT/RPadExtent.hxx>
#include <ROOT/RPadPos.hxx>

namespace ROOT {
namespace Experimental {

/** class ROOT::Experimental::RAttrBox
 Drawing attributes for a box: rectangular lines with size and position.
 */
class RAttrBox : public RAttributesVisitor {
   RAttrLine fBorder{this, "border_"};     ///<!
   RAttrLine fTop{this, "top_"};           ///<!
   RAttrLine fRight{this, "right_"};       ///<!
   RAttrLine fBottom{this, "bottom_"};     ///<!
   RAttrLine fLeft{this, "left_"};         ///<!

protected:
   const RDrawableAttributes::Map_t &GetDefaults() const override
   {
      static auto dflts = RDrawableAttributes::Map_t().AddDefaults(fBorder).AddDefaults(fTop).AddDefaults(fRight).AddDefaults(fBottom).AddDefaults(fLeft);
      return dflts;
   }

public:
   using RAttributesVisitor::RAttributesVisitor;

   const RAttrLine &Border() const { return fBorder; }
   RAttrLine &Border() { return fBorder; }
   /// Overrides Border() for the top line..
   const RAttrLine &Top() const { return fTop; }
   RAttrLine &Top() { return fTop; }
   /// Overrides Border() for the right line.
   const RAttrLine &Right() const { return fRight; }
   RAttrLine &Right() { return fRight; }
   /// Overrides Border() for the bottom line.
   const RAttrLine &Bottom() const { return fBottom; }
   RAttrLine &Bottom() { return fBottom; }
   /// Overrides Border() for the left line.
   const RAttrLine &Left() const { return fLeft; }
   RAttrLine &Left() { return fLeft; }

   // TODO: Add Fill()!
};

} // namespace Experimental
} // namespace ROOT

#endif
