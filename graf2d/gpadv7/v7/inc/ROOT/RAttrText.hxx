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

#include <ROOT/RDrawingAttr.hxx>
#include <ROOT/RColor.hxx>

#include <string>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::RText
 A text.
 */

class RAttrText: public RDrawingAttrBase {
public:

   using RDrawingAttrBase::RDrawingAttrBase;

   /// The color of the text.
   RAttrText &SetColor(const RColor &col) { Set("color", col); return *this; }
   RColor GetColor() const { return Get<RColor>("color"); }

   /// The size of the text.
   RAttrText &SetSize(float size) { Set("size", size); return *this; }
   float GetSize() const { return Get<float>("size"); }

   /// The angle of the text.
   RAttrText &SetAngle(float angle) { Set("angle", angle); return *this; }
   float GetAngle() const { return Get<float>("angle"); }

   /// The alignment of the text.
   RAttrText &SetAlign(int style) { Set("align", style); return *this; }
   int GetAlign() const { return Get<int>("align"); }

   /// The font of the text.
   RAttrText &SetFont(int font) { Set("font", font); return *this; }
   int GetFont() const { return Get<int>("font"); }
};

} // namespace Experimental
} // namespace ROOT

#endif
