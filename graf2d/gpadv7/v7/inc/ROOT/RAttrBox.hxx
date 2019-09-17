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
#include <ROOT/RAttrFill.hxx>
#include <ROOT/RPadExtent.hxx>
#include <ROOT/RPadPos.hxx>

namespace ROOT {
namespace Experimental {

/** class ROOT::Experimental::RAttrBox
 Drawing attributes for a box: rectangular lines with size and position.
 */
class RAttrBox : public RAttrBase {

   RAttrLine fBorder{this, "border_"};       ///<!
   RAttrFill fFill{this, "fill_"};           ///<!

protected:
   const RDrawingAttr::Map_t &GetDefaults() const override
   {
      static auto dflts = RDrawingAttr::Map_t().AddDefaults(fBorder).AddDefaults(fFill);
      return dflts;
   }

public:

   using RAttrBase::RAttrBase;

   RAttrBox(const RAttrBox &src) : RAttrBox() { src.CopyTo(*this); }
   RAttrBox &operator=(const RAttrBox &src) { Clear(); src.CopyTo(*this); return *this; }

   const RAttrLine &Border() const { return fBorder; }
   RAttrLine &Border() { return fBorder; }

   const RAttrFill &Fill() const { return fFill; }
   RAttrFill &Fill() { return fFill; }
};

} // namespace Experimental
} // namespace ROOT

#endif
