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

#include <ROOT/RDrawingAttr.hxx>
#include <ROOT/RColor.hxx>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::RMarker
 A simple marker.
 */

class RAttrMarker : public RDrawingAttrBase {
   /// The color of the marker.
   RColor fColor;

   /// The size of the marker.
   float fSize = 3.;

   /// The style of the marker.
   int fStyle = 0;

private:
   std::vector<MemberAssociation> GetMembers() final {
      return {
         Associate("color", fColor),
         Associate("size", fSize),
         Associate("style", fStyle)
      };
   };

public:
   std::unique_ptr<RDrawingAttrBase> Clone() const { return std::make_unique<RAttrMarker>(*this); }

   /// The color of the marker.
   RAttrMarker &SetColor(const RColor &col) { fColor = col; return *this; }
   const RColor &GetColor() const { return fColor; }

   /// The size of the marker.
   RAttrMarker &SetSize(float size) { fSize = size; return *this; }
   float GetSize() const { return fSize; }

   /// The style of the marker.
   RAttrMarker &SetStyle(int style) { fStyle = style; return *this; }
   int GetStyle() const { return fStyle; }

   bool operator==(const RAttrMarker &other) const {
      return fColor == other.fColor && fSize == other.fSize && fStyle == other.fStyle;
   }

   bool operator!=(const RAttrMarker &other) const {
      return !(*this == other);
   }
};

} // namespace Experimental
} // namespace ROOT

#endif
