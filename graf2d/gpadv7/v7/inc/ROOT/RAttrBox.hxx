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
class RAttrBox: public RDrawingAttrBase {
   RAttrLine fBorder; /// Line style, can be overriden for each side.
   RAttrLine fTop; /// Overrides Border() for the top line.
   RAttrLine fRight; /// Overrides Border() for the right line.
   RAttrLine fBottom; /// Overrides Border() for the bottom line.
   RAttrLine fLeft; /// Overrides Border() for the left line.

   std::vector<MemberAssociation> GetMembers() final {
      return {
         Associate("border", fBorder),
         Associate("top", fTop),
         Associate("right", fRight),
         Associate("bottom", fBottom),
         Associate("left", fLeft)
      };
   }

public:

   RAttrLine &Border() { return  fBorder; }
   /// Overrides Border() for the top line.
   RAttrLine &Top() { return fTop; }
   /// Overrides Border() for the right line.
   RAttrLine &Right() { return fRight; }
   /// Overrides Border() for the bottom line.
   RAttrLine &Bottom() { return fBottom; }
   /// Overrides Border() for the left line.
   RAttrLine &Left() { return fLeft; }

   // TODO: Add Fill()!

   bool operator==(const RAttrBox &other) const {
      return fBorder == other.fBorder
         && fTop == other.fTop
         && fRight == other.fRight
         && fBottom == other.fBottom
         && fLeft == other.fLeft;
   }

   bool operator!=(const RAttrBox &other) const {
      return !(*this == other);
   }
};

} // namespace Experimental
} // namespace ROOT

#endif
