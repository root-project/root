/// \file ROOT/RAttrText.hxx
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

#ifndef ROOT7_RAttrText
#define ROOT7_RAttrText

#include <ROOT/RColor.hxx>
#include <ROOT/RDrawingAttr.hxx>

#include <string>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::RText
 A text.
 */

class RAttrText: public RDrawingAttrBase {
public:

   RAttrText() = default;
   RAttrText(const char* name, RDrawingAttrHolderBase* holder, RDrawingAttrBase *parent = nullptr):
      RDrawingAttrBase(name, holder, parent, {"color", "size", "angle", "align", "font"})
   {}

public:
   /// The color of the text.
   void SetColor(const RColor &col) { Set(0, ColorToString(col)); }
   std::pair<RColor, bool> GetColor() const {
      auto ret = Get(0);
      return {ColorFromString("text color", ret.first), ret.second};
   }

   /// The size of the text.
   void SetSize(float size) { Set(1, std::to_string(size)); }
   std::pair<float, bool> GetSize() const {
      auto ret = Get(1);
      return {std::stof(ret.first), ret.second};
   }

   /// The angle of the text.
   void SetAngle(float angle) { Set(2, std::to_string(angle)); }
   std::pair<float, bool> GetAngle() const {
      auto ret = Get(2);
      return {std::stof(ret.first), ret.second};
   }

   /// The alignment of the text.
   void SetAlign(int style) { Set(3, std::to_string(style)); }
   std::pair<int, bool> GetAlign() const {
      auto ret = Get(3);
      return {std::stoi(ret.first), ret.second};
   }

   /// The font of the text.
   void SetFont(int font) { Set(4, std::to_string(font)); }
   std::pair<int, bool> GetFont() const {
      auto ret = Get(4);
      return {std::stoi(ret.first), ret.second};
   }

};

} // namespace Experimental
} // namespace ROOT

#endif
