/// \file ROOT/RAttrText.hxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2018-10-12
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RAttrText
#define ROOT7_RAttrText

#include <ROOT/RAttrBase.hxx>
#include <ROOT/RColor.hxx>

#include <string>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::RText
 A text.
 */


class RAttrText : public RAttrBase {

   RColor fColor{this, "color_"}; ///<! line color, will access container from line attributes

   R__ATTR_CLASS(RAttrText, "text_", AddDouble("size", 12.).AddDouble("angle", 0.).AddInt("align", 22).AddInt("font", 41).AddDefaults(fColor));

   ///The text size
   RAttrText &SetSize(double width) { SetValue("size", width); return *this; }
   double GetSize() const { return GetValue<double>("size"); }

   ///The text angle
   RAttrText &SetAngle(double angle) { SetValue("angle", angle); return *this; }
   double GetAngle() const { return GetValue<double>("angle"); }

   ///The text alignment
   RAttrText &SetAlign(int align) { SetValue("align", align); return *this; }
   int GetAlign() const { return GetValue<int>("align"); }

   ///The text font
   RAttrText &SetFont(int font) { SetValue("font", font); return *this; }
   int GetFont() const { return GetValue<int>("font"); }

   ///The color of the text.
   RAttrText &SetColor(const RColor &color) { fColor = color; return *this; }
   const RColor &Color() const { return fColor; }
   RColor &Color() { return fColor; }

};



} // namespace Experimental
} // namespace ROOT

#endif
