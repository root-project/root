/// \file ROOT/RAttrBox.hxx
/// \ingroup Graf ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2018-10-17
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RAttrBox
#define ROOT7_RAttrBox

#include <ROOT/RAttrLine.hxx>
#include <ROOT/RColor.hxx>
#include <ROOT/RDrawingAttr.hxx>
#include <ROOT/RPadExtent.hxx>
#include <ROOT/RPadPos.hxx>

namespace ROOT {
namespace Experimental {

/** class ROOT::Experimental::RAttrBox
 Drawing attributes for a box: rectangular lines with size and position.
 */
class RAttrBox: public RDrawingAttrBase {
public:
   using RDrawingAttrBase::RDrawingAttrBase;

   RAttrLine Border() const { return  {"border", *this}; }
   /// Overrides Border() for the top line..
   RAttrLine Top() const { return  {"top", *this}; }
   /// Overrides Border() for the right line.
   RAttrLine Right() const { return  {"right", *this}; }
   /// Overrides Border() for the bottom line.
   RAttrLine Bottom() const { return  {"bottom", *this}; }
   /// Overrides Border() for the left line.
   RAttrLine Left() const { return  {"left", *this}; }
   
   // TODO: Add Fill()!
};

} // namespace Experimental
} // namespace ROOT

#endif
