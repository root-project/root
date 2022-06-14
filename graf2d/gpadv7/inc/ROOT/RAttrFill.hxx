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

   enum EStyle {
      kHollow = 0,
      kNone = 0,
      kSolid = 1001,
      k3001 = 3001, k3002 = 3002, k3003 = 3003, k3004 = 3004, k3005 = 3005,
      k3006 = 3006, k3007 = 3007, k3008 = 3008, k3009 = 3009, k3010 = 3010,
      k3011 = 3011, k3012 = 3012, k3013 = 3013, k3014 = 3014, k3015 = 3015,
      k3016 = 3016, k3017 = 3017, k3018 = 3018, k3019 = 3019, k3020 = 3020,
      k3021 = 3021, k3022 = 3022, k3023 = 3023, k3024 = 3024, k3025 = 3025
   };

   RAttrValue<RColor> color{this, "color", RColor::kBlack};  ///<! fill color
   RAttrValue<EStyle> style{this, "style", kHollow};         ///<! fill style

   RAttrFill(RColor _color, EStyle _style) : RAttrFill()
   {
      color = _color;
      style = _style;
   }
};

} // namespace Experimental
} // namespace ROOT

#endif
