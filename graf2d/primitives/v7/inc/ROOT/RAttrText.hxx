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

   using RDrawingAttrBase::RDrawingAttrBase;

   /// The color of the text.
   void SetColor(const RColor &col) { Set("color", col); }
   RColor GetColor() const { return Get<RColor>("color"); }

   /// The size of the text.
   void SetSize(float size) { Set("size", size); }
   float GetSize() const { return Get<float>("size"); }

   /// The angle of the text.
   void SetAngle(float angle) { Set("angle", angle); }
   float GetAngle() const { return Get<float>("angle"); }

   /// The alignment of the text.
   void SetAlign(int style) { Set("align", style); }
   int GetAlign() const { return Get<int>("align"); }

   /// The font of the text.
   void SetFont(int font) { Set("font", font); }
   int GetFont() const { return Get<int>("font"); }
};

} // namespace Experimental
} // namespace ROOT

#endif
