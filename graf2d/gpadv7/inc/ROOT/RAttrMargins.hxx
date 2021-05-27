/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RAttrMargins
#define ROOT7_RAttrMargins

#include <ROOT/RAttrBase.hxx>
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

class RAttrMargins : public RAttrBase {

   RAttrValue<RPadLength>   fMarginLeft{"left", this, 0._normal};
   RAttrValue<RPadLength>   fMarginRight{"right", this, 0._normal};
   RAttrValue<RPadLength>   fMarginTop{"top", this, 0._normal};
   RAttrValue<RPadLength>   fMarginBottom{"bottom", this, 0._normal};
   RAttrValue<RPadLength>   fMarginAll{"all", this, 0._normal};

   R__ATTR_CLASS(RAttrMargins, "margin");

public:

   RAttrMargins &MarginSetLeft(const RPadLength &len) { fMarginLeft = len; return *this; }
   RPadLength GetMarginLeft() const { return fMarginLeft; }

   RAttrMargins &SetMarginRight(const RPadLength &len) { fMarginRight = len; return *this; }
   RPadLength GetMarginRight() const { return fMarginRight; }

   RAttrMargins &SetMarginTop(const RPadLength &len) { fMarginTop = len; return *this; }
   RPadLength GetMarginTop() const { return fMarginTop; }

   RAttrMargins &SetMarginBottom(const RPadLength &len) { fMarginBottom = len; return *this; }
   RPadLength GetMarginBottom() const { return fMarginBottom; }

   RAttrMargins &SetMarginAll(const RPadLength &len) { fMarginAll = len; return *this; }
   RPadLength GetMarginAll() const { return fMarginAll; }

};

} // namespace Experimental
} // namespace ROOT

#endif
