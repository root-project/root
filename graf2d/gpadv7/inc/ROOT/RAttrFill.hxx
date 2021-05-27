/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RAttrFill
#define ROOT7_RAttrFill

#include <ROOT/RAttrBase.hxx>
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

class RAttrFill : public RAttrBase {

   RAttrValue<RColor> fFillColor{"color", this, RColor::kBlack};  ///<! fill color
   RAttrValue<int>    fFillStyle{"style", this, 1};               ///<! fill style

   R__ATTR_CLASS(RAttrFill, "fill");

   ///The fill style
   RAttrFill &SetFillStyle(int style) { fFillStyle = style; return *this; }
   int GetFillStyle() const { return fFillStyle; }

   ///The fill color
   RAttrFill &SetFillColor(const RColor &color) { fFillColor = color; return *this; }
   RColor GetFillColor() const { return fFillColor; }

   const RAttrFill &AttrFill() const { return *this; }
   RAttrFill &AttrFill() { return *this; }

};

} // namespace Experimental
} // namespace ROOT

#endif
