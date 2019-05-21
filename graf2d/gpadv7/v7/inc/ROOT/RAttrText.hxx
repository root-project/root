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
   /// The color of the text.
   RColor fColor;

   /// The size of the text.
   float fSize = 1.;

   /// The angle of the text.
   float fAngle = 0.;

   /// The alignment of the text.
   int fAlign = 0;

   /// The font of the text.
   int fFont = 0;

private:
   std::vector<MemberAssociation> GetMembers() final {
      return {
         Associate("color", fColor),
         Associate("size", fSize),
         Associate("angle", fAngle),
         Associate("align", fAlign),
         Associate("font", fFont)
      };
   }

public:

   using RDrawingAttrBase::RDrawingAttrBase;

   /// The color of the text.
   RAttrText &SetColor(const RColor &col) { fColor = col; return *this; }
   const RColor &GetColor() const { return fColor; }

   /// The size of the text.
   RAttrText &SetSize(float size) { fSize = size; return *this; }
   float GetSize() const { return fSize; }

   /// The angle of the text.
   RAttrText &SetAngle(float angle) { fAngle = angle; return *this; }
   float GetAngle() const { return fAngle; }

   /// The alignment of the text.
   RAttrText &SetAlign(int align) { fAlign = align; return *this; }
   int GetAlign() const { return fAlign; }

   /// The font of the text.
   RAttrText &SetFont(int font) { fFont = font; return *this; }
   int GetFont() const { return fFont; }

   bool operator==(const RAttrText &other) const {
      return fColor == other.fColor
         && fSize == other.fSize
         && fAngle == other.fAngle
         && fAlign == other.fAlign
         && fFont == other.fFont;
   }

   bool operator!=(const RAttrText &other) const {
      return !(*this == other);
   }
};

} // namespace Experimental
} // namespace ROOT

#endif
