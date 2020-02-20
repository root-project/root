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

class RAttrLength : public RAttrBase {

   RPadLength fLength;      ///<! current position

   R__ATTR_CLASS(RAttrLength, "shift_", AddDouble("norm", 0.).AddDouble("px", 0));

   RAttrLength &Set(const RPadLength &pos)
   {
      if (pos.HasNormal())
         SetValue("norm", pos.GetNormal());
      else
         ClearValue("norm");

      if (pos.HasPixel())
         SetValue("px", pos.GetPixel());
      else
         ClearValue("px");

      return *this;
   }

   RPadLength Get() const
   {
      RPadLength res;

      auto norm = GetValue<double>("norm");
      auto px = GetValue<double>("px");

      if (px) res.SetPixel(px);
      if (norm) res.SetNormal(norm);

      return res;
   }

};

class RAttrMargins : public RAttrBase {

   RAttrLength fLeft{this, "left_"}; ///<! left margin
   RAttrLength fRight{this, "right_"}; ///<! right margin
   RAttrLength fTop{this, "top_"}; ///<! top margin
   RAttrLength fBottom{this, "bottom_"}; ///<! bottom margin

   R__ATTR_CLASS(RAttrMargins, "margin_", AddDefaults(fLeft).AddDefaults(fRight).AddDefaults(fTop).AddDefaults(fBottom));

   RAttrMargins &SetLeft(const RPadLength &pos) { fLeft.Set(pos); return *this; }
   RPadLength GetLeft() const { return fLeft.Get(); }
   RAttrLength &Left() { return fLeft; }

   RAttrMargins &SetRight(const RPadLength &pos) { fRight.Set(pos); return *this; }
   RPadLength GetRight() const { return fRight.Get(); }
   RAttrLength &Right() { return fRight; }

   RAttrMargins &SetTop(const RPadLength &pos) { fTop.Set(pos); return *this; }
   RPadLength GetTop() const { return fTop.Get(); }
   RAttrLength &Top() { return fTop; }

   RAttrMargins &SetBottom(const RPadLength &pos) { fBottom.Set(pos); return *this; }
   RPadLength GetBottom() const { return fBottom.Get(); }
   RAttrLength &Bottom() { return fBottom; }

};

} // namespace Experimental
} // namespace ROOT

#endif
