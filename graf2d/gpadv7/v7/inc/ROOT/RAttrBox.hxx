/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RAttrBox
#define ROOT7_RAttrBox

#include <ROOT/RAttrBase.hxx>
#include <ROOT/RAttrLine.hxx>
#include <ROOT/RAttrFill.hxx>

namespace ROOT {
namespace Experimental {

/** \class RAttrBox
\ingroup GpadROOT7
\author Axel Naumann <axel@cern.ch>
\date 2018-10-17
\brief Drawing attributes for a box: rectangular lines with size and position.
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RAttrBox : public RAttrBase {

   RAttrLine fBorder{this, "border_"};       ///<!
   RAttrFill fFill{this, "fill_"};           ///<!

   R__ATTR_CLASS(RAttrBox, "box_", AddDefaults(fBorder).AddDefaults(fFill));

   const RAttrLine &Border() const { return fBorder; }
   RAttrLine &Border() { return fBorder; }

   const RAttrFill &Fill() const { return fFill; }
   RAttrFill &Fill() { return fFill; }
};

} // namespace Experimental
} // namespace ROOT

#endif
