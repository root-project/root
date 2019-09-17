/// \file ROOT/RAttrMarker.hxx
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

#ifndef ROOT7_RAttrMarker
#define ROOT7_RAttrMarker

#include <ROOT/RAttrBase.hxx>
#include <ROOT/RColor.hxx>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::RMarker
 A simple marker.
 */

class RAttrMarker : public RAttrBase {

   RColor fColor{this, "color_"}; ///<! marker color, will access container from line attributes

protected:
   const RAttrValues::Map_t &GetDefaults() const override
   {
      static auto dflts = RAttrValues::Map_t().AddDouble("size",1.).AddInt("style",1).AddDefaults(fColor);
      return dflts;
   }

public:
   using RAttrBase::RAttrBase;

   RAttrMarker(const RAttrMarker &src) : RAttrMarker() { src.CopyTo(*this); }
   RAttrMarker &operator=(const RAttrMarker &src) { Clear(); src.CopyTo(*this); return *this; }

   RAttrMarker &SetColor(const RColor &color) { fColor = color; return *this; }
   const RColor &Color() const { return fColor; }
   RColor &Color() { return fColor; }

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
