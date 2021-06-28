/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RAttrMargins
#define ROOT7_RAttrMargins

#include <ROOT/RAttrAggregation.hxx>
#include <ROOT/RAttrValue.hxx>
#include <ROOT/RPadLength.hxx>

namespace ROOT {
namespace Experimental {

/** \class RAttrMargins
\ingroup GpadROOT7
\author Sergey Linev <s.linev@gsi.de>
\date 2020-02-20
\brief A margins attributes. Only relative and pixel coordinates are allowed
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RAttrMargins : public RAttrAggregation {

   R__ATTR_CLASS(RAttrMargins, "margins");

   RAttrValue<RPadLength> left{this, "left", 0._normal};     ///<! left margin
   RAttrValue<RPadLength> right{this, "right", 0._normal};   ///<! right margin
   RAttrValue<RPadLength> top{this, "top", 0._normal};       ///<! top margin
   RAttrValue<RPadLength> bottom{this, "bottom", 0._normal}; ///<! bottom margin

   RAttrMargins &operator=(const RPadLength &len)
   {
      left = len;
      right = len;
      top = len;
      bottom = len;
      return *this;
   }
};

} // namespace Experimental
} // namespace ROOT

#endif
