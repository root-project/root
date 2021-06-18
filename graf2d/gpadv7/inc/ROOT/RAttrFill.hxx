/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RAttrFill
#define ROOT7_RAttrFill

#include <ROOT/RAttrAggregation.hxx>
#include <ROOT/RAttrValue.hxx>

namespace ROOT {
namespace Experimental {

/** \class RAttrFill
\ingroup GpadROOT7
\author Sergey Linev
\date 2019-09-13
\brief Drawing fill attributes for different objects.
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RAttrFill : public RAttrAggregation {

   RAttrValue<RColor> fColor{this, "color", RColor::kBlack};  ///<! fill color
   RAttrValue<int>    fStyle{this, "style", 1};               ///<! fill style

   R__ATTR_CLASS(RAttrFill, "fill");

   ///The fill style
   RAttrFill &SetStyle(int style) { fStyle = style; return *this; }
   int GetStyle() const { return fStyle; }

   ///The fill color
   RAttrFill &SetColor(const RColor &color) { fColor = color; return *this; }
   RColor GetColor() const { return fColor; }

};

} // namespace Experimental
} // namespace ROOT

#endif
