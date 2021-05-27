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
#include <ROOT/RAttrValue.hxx>

namespace ROOT {
namespace Experimental {

/** \class RAttrMarker
\ingroup GpadROOT7
\authors Axel Naumann <axel@cern.ch> Sergey Linev <s.linev@gsi.de>
\date 2018-10-12
\brief A marker attributes.
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RAttrMarker : public RAttrBase {

   RAttrValue<RColor>   fMarkerColor{"color", this, RColor::kBlack};  ///<! marker color
   RAttrValue<double>   fMarkerSize{"size", this, 1.};                ///<! marker size
   RAttrValue<int>      fMarkerStyle{"style", this, 1};               ///<! marker style

   R__ATTR_CLASS(RAttrMarker, "marker");

   RAttrMarker &SetMarkerColor(const RColor &color) { fMarkerColor = color; return *this; }
   RColor GetMarkerColor() const { return fMarkerColor; }

   /// The size of the marker.
   RAttrMarker &SetMarkerSize(double size) { fMarkerSize = size; return *this; }
   double GetMarkerSize() const { return fMarkerSize; }

   /// The style of the marker.
   RAttrMarker &SetMarkerStyle(int style) { fMarkerStyle = style; return *this; }
   int GetMarkerStyle() const { return fMarkerStyle; }

   const RAttrMarker &AttrMarker() const { return *this; }
   RAttrMarker &AttrMarker() { return *this; }

};

} // namespace Experimental
} // namespace ROOT

#endif
