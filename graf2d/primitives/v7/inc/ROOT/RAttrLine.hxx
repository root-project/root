/// \file ROOT/RAttrLine.hxx
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

#ifndef ROOT7_RAttrLine
#define ROOT7_RAttrLine

#include <ROOT/RColor.hxx>
#include <ROOT/RDrawingAttr.hxx>

namespace ROOT {
namespace Experimental {

/** class ROOT::Experimental::RAttrLine
 Drawing attributes for RLine.
 */
class RAttrLine: public RDrawingAttrBase {
public:
   RAttrLine() = default;

   RAttrLine(const char* name, RDrawingAttrHolderBase* holder, RDrawingAttrBase *parent = nullptr):
      RDrawingAttrBase(name, holder, parent, {"color", "width", "style"})
   {}

   /// The color of the line.
   void SetColor(const RColor& col) { Set(0, ColorToString(col)); }
   std::pair<RColor, bool> GetColor() const
   {
      auto ret = Get(0);
      return {ColorFromString("line color", ret.first), ret.second};
   }

   ///The width of the line.
   void SetWidth(float width) { Set(1, std::to_string(width)); }
   std::pair<float, bool> GetWidth() const {
      auto ret = Get(1);
      return {std::stof(ret.first), ret.second};
   }

   ///The style of the line.
   void SetStyle(int style) { Set(2, std::to_string(style)); }
   std::pair<int, bool> GetStyle() const {
      auto ret = Get(2);
      return {std::stoi(ret.first), ret.second};
   }

};

} // namespace Experimental
} // namespace ROOT

#endif
