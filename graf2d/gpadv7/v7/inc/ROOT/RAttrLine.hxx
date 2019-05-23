/// \file ROOT/RAttrLine.hxx
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

#ifndef ROOT7_RAttrLine
#define ROOT7_RAttrLine

#include <ROOT/RDrawingAttr.hxx>
#include <ROOT/RColor.hxx>

namespace ROOT {
namespace Experimental {

/** class ROOT::Experimental::RAttrLine
 Drawing attributes for RLine.
 */
class RAttrLine: public RDrawingAttrBase {
   RColor fColor; /// The color of the line.
   float fWidth = 3; ///The width of the line.
   int fStyle = 0; ///The style of the line.

protected:
   std::vector<MemberAssociation> GetMembers() final {
      return {
         Associate("color", fColor),
         Associate("width", fWidth),
         Associate("style", fStyle)};
   }

public:
   std::unique_ptr<RDrawingAttrBase> Clone() const { return std::make_unique<RAttrLine>(*this); }

   /// The color of the line.
   RAttrLine &SetColor(const RColor& col) { fColor = col; return *this; }
   RColor GetColor() const { return fColor; }

   ///The width of the line.
   RAttrLine &SetWidth(float width) { fWidth = width; return *this; }
   float GetWidth() const { return fWidth; }

   ///The style of the line.
   RAttrLine &SetStyle(int style) { fStyle = style; return *this; }
   int GetStyle() const { return fStyle; }

   bool operator==(const RAttrLine &other) const {
      return fColor == other.fColor
         && fWidth == other.fWidth
         && fStyle == other.fStyle;
   }

   bool operator!=(const RAttrLine &other) const {
      return !(*this == other);
   }
};

} // namespace Experimental
} // namespace ROOT

#endif
