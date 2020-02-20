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

   R__ATTR_CLASS(RAttrMargins, "margin_", AddString("left","").AddString("right","").AddString("top","").AddString("bottom",""));

   RAttrMargins &SetLeft(const RPadLength &pos) { return SetMargin("left", pos); }
   RPadLength GetLeft() const { return GetMargin("left"); }

   RAttrMargins &SetRight(const RPadLength &pos) { return SetMargin("right", pos); }
   RPadLength GetRight() const { return GetMargin("right"); }

   RAttrMargins &SetTop(const RPadLength &pos) { return SetMargin("top", pos); }
   RPadLength GetTop() const { return GetMargin("top"); }

   RAttrMargins &SetBottom(const RPadLength &pos) { return SetMargin("bottom", pos); }
   RPadLength GetBottom() const { return GetMargin("bottom"); }

protected:

   RAttrMargins &SetMargin(const std::string &name, const RPadLength &pos)
   {
      if (pos.Empty())
         ClearValue(name);
      else
         SetValue(name, pos.AsString());

      return *this;
   }

   RPadLength GetMargin(const std::string &name) const
   {
      RPadLength res;

      auto value = GetValue<std::string>(name);

      if (!value.empty())
         res.ParseString(value);

      return res;
   }

};

} // namespace Experimental
} // namespace ROOT

#endif
