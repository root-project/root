/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RAttrText
#define ROOT7_RAttrText

#include <ROOT/RAttrAggregation.hxx>
#include <ROOT/RAttrValue.hxx>
#include <ROOT/RAttrFont.hxx>

namespace ROOT {
namespace Experimental {

/** \class RAttrText
\ingroup GpadROOT7
\brief A text attributes.
\authors Axel Naumann <axel@cern.ch> Sergey Linev <s.linev@gsi.de>
\date 2018-10-12
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RAttrText : public RAttrAggregation {

   R__ATTR_CLASS(RAttrText, "text");

public:

   enum EAlign {
      kLeftBottom = 11,
      kLeftCenter = 12,
      kLeftTop = 13,
      kCenterBottom = 21,
      kCenter = 22,
      kCenterTop = 23,
      kRightBottom = 31,
      kRightCenter = 32,
      kRightTop = 33
   };

   RAttrValue<RColor> color{this, "color", RColor::kBlack};  ///<! text color
   RAttrValue<double> size{this, "size", 12.};               ///<! text size
   RAttrValue<double> angle{this, "angle", 0.};              ///<! text angle
   RAttrValue<EAlign> align{this, "align", kCenter};         ///<! text align
   RAttrFont font{this, "font"};                             ///<! text font

   RAttrText(RDrawable *drawable, const char *prefix, double _size) : RAttrAggregation(drawable, prefix), size(this, "size", _size) {}
};

} // namespace Experimental
} // namespace ROOT

#endif
