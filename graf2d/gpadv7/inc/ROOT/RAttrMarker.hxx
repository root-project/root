/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RAttrMarker
#define ROOT7_RAttrMarker

#include <ROOT/RAttrBase.hxx>
#include <ROOT/RAttrColor.hxx>

namespace ROOT {
namespace Experimental {

/** \class RAttrMarker
\ingroup GpadROOT7
\author Axel Naumann <axel@cern.ch>
\date 2018-10-12
\brief A marker attributes.
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RAttrMarker : public RAttrBase {

   RAttrColor fColor{this, "color_"}; ///<! marker color, will access container from line attributes

   R__ATTR_CLASS(RAttrMarker, "marker_", AddDouble("size", 1.).AddInt("style", 1).AddDefaults(fColor));

   RAttrMarker &SetColor(const RColor &color) { fColor = color; return *this; }
   RColor GetColor() const { return fColor.GetColor(); }
   RAttrColor &AttrColor() { return fColor; }

   /// The size of the marker.
   RAttrMarker &SetSize(float size) { SetValue("size", size); return *this; }
   float GetSize() const { return GetValue<double>("size"); }

   /// The style of the marker.
   RAttrMarker &SetStyle(int style) { SetValue("style", style); return *this; }
   int GetStyle() const { return GetValue<int>("style"); }
};

} // namespace Experimental
} // namespace ROOT

#endif
