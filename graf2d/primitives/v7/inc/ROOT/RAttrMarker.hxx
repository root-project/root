/// \file ROOT/RAttrMarker.hxx
/// \ingroup Graf ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2018-10-12
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RAttrMarker
#define ROOT7_RAttrMarker

#include <ROOT/RColor.hxx>
#include <ROOT/RDrawingAttr.hxx>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::RMarker
 A simple marker.
 */

class RAttrMarker : public RDrawingAttrBase {
public:
   RAttrMarker() = default;
   RAttrMarker(const char* name, RDrawingAttrHolderBase* holder, RDrawingAttrBase *parent = nullptr):
      RDrawingAttrBase(name, holder, parent, {"color", "size", "style"})
   {}

public:
   /// The color of the marker.
   void SetColor(const RColor &col) { Set(0, ColorToString(col)); }
   std::pair<RColor, bool> GetColor() const {
      auto ret = Get(0);
      return {ColorFromString("marker color", ret.first), ret.second};
   }

   /// The size of the marker.
   void SetSize(float size) { Set(1, std::to_string(size)); }
   std::pair<float, bool> GetSize() const {
      auto ret = Get(1);
      return {std::stof(ret.first), ret.second};
   }

   /// The style of the marker.
   void SetStyle(int style) { Set(2, std::to_string(style)); }
   std::pair<int, bool> GetStyle() const {
      auto ret = Get(2);
      return {std::stoi(ret.first), ret.second};
   }
};

} // namespace Experimental
} // namespace ROOT

#endif
