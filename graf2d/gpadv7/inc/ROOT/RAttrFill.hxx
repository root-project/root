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

   R__ATTR_CLASS(RAttrFill, "fill");

public:

   RAttrValue<RColor>  color{this, "color", RColor::kBlack};  ///<! fill color
   RAttrValue<int>     style{this, "style", 1};               ///<! fill style

   RAttrFill(RColor _color, int _style) : RAttrFill()
   {
      color = _color;
      style = _style;
   }
};

} // namespace Experimental
} // namespace ROOT

#endif
