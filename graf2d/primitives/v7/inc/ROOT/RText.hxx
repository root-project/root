/// \file ROOT/RText.hxx
/// \ingroup Graf ROOT7
/// \author Olivier Couet <Olivier.Couet@cern.ch>
/// \date 2017-10-16
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RText
#define ROOT7_RText

#include <ROOT/RDrawable.hxx>
#include <ROOT/RDrawingAttr.hxx>
#include <ROOT/RDrawingOptsBase.hxx>
#include <ROOT/RPadPos.hxx>
#include <ROOT/RPadPainter.hxx>

#include <initializer_list>
#include <memory>
#include <string>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::RText
 A text.
 */

class RText : public RDrawableBase<RText> {
public:

/** class ROOT::Experimental::RText::DrawingOpts
 Drawing options for RText.
 */

class DrawingOpts: public RDrawingOptsBase {
   RDrawingAttr<RColor> fTextColor{*this, "Text.Color", RColor::kBlack}; ///< The text color.
   RDrawingAttr<float>  fTextSize{*this, "Text.Size", 10.};              ///< The text size.
   RDrawingAttr<float>  fTextAngle{*this, "Text.Angle", 0.};             ///< The text angle.
   RDrawingAttr<int>    fTextAlign{*this, "Text.Align", 13.};            ///< The text align.
   RDrawingAttr<int>    fTextFont{*this, "Text.Font", 42.};              ///< The text font.


public:
   /// The color of the text.
   void SetTextColor(const RColor &col) { fTextColor = col; }
   RDrawingAttr<RColor> &GetTextColor() { return fTextColor; }
   const RColor &GetTextColor() const   { return fTextColor.Get(); }

   /// The text size.
   void SetTextSize(float size) { fTextSize = size; }
   RDrawingAttr<float> &GetTextSize() { return fTextSize; }
   float GetTextSize() const { return (float)fTextSize; }

   /// The text angle in degrees.
   void SetTextAngle(float angle) { fTextAngle = angle; }
   RDrawingAttr<float> &GetTextAngle() { return fTextAngle; }
   float GetTextAngle() const { return (float)fTextAngle; }

   ///The text align.
   void SetTextAlign(int align) { fTextAlign = align; }
   RDrawingAttr<int> &GetTextAlign() {return fTextAlign; }
   int GetTextAlign() const {return (int) fTextAlign; }

   ///The text font.
   void SetTextFont(int font) {fTextFont = font; }
   RDrawingAttr<int> &GetTextFont() {return fTextFont; }
   int GetTextFont() const {return (int) fTextFont; }

};


private:

   /// The text itself
   std::string fText;

   /// Text's position
   RPadPos fP;

   /// Text's attributes
   DrawingOpts fOpts;

public:

   RText() = default;

   RText(const std::string &str) : fText(str) {}
   RText(const RPadPos& p, const std::string &str) : fText(str), fP(p) {}

   void SetText(const std::string &txt) { fText = txt; }

   std::string GetText() const { return fText; }

   void SetPosition(const RPadPos& p) {fP = p;}

   const RPadPos& GetPosition() const { return fP; }


   /// Get the drawing options.
   DrawingOpts &GetOptions() { return fOpts; }
   const DrawingOpts &GetOptions() const { return fOpts; }

   void Paint(Internal::RPadPainter &pad) final
   {
      pad.AddDisplayItem(
         std::make_unique<ROOT::Experimental::ROrdinaryDisplayItem<ROOT::Experimental::RText>>(this));
   }
};

inline std::shared_ptr<ROOT::Experimental::RText>
GetDrawable(const std::shared_ptr<ROOT::Experimental::RText> &text)
{
   /// A RText is a RDrawable itself.
   return text;
}

} // namespace Experimental
} // namespace ROOT

#endif
