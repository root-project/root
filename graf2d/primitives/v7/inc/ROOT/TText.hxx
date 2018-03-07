/// \file ROOT/TText.hxx
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

#ifndef ROOT7_TText
#define ROOT7_TText

#include <ROOT/TDrawable.hxx>
#include <ROOT/TDrawingAttr.hxx>
#include <ROOT/TDrawingOptsBase.hxx>
#include <ROOT/TPad.hxx>
#include <ROOT/TVirtualCanvasPainter.hxx>

#include <initializer_list>
#include <memory>
#include <string>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::TText
 A text.
 */

class TText : public TDrawableBase<TText> {
public:

/** class ROOT::Experimental::TText::DrawingOpts
 Drawing options for TText.
 */

class DrawingOpts: public TDrawingOptsBase {
   TDrawingAttr<TColor> fTextColor{*this, "Text.Color", TColor::kBlack};   ///< The text color.
   TDrawingAttr<int>    fTextSize{*this, "Text.Size", 10};                 ///< The text size
   TDrawingAttr<int>    fTextAngle{*this, "Text.Angle", 0};                ///< The text angle

public:
   /// The color of the text.
   void SetTextColor(const TColor &col) { fTextColor = col; }
   TDrawingAttr<TColor> &GetTextColor() { return fTextColor; }
   const TColor &GetTextColor() const   { return fTextColor.Get(); }

   /// The text size.
   void SetTextSize(int size) { fTextSize = size; }
   TDrawingAttr<int> &GetTextSize() { return fTextSize; }
   int GetTextSize() const { return (int)fTextSize; }

   /// The text angle in degrees.
   void SetTextAngle(int angle) { fTextAngle = angle; }
   TDrawingAttr<int> &GetTextAngle() { return fTextAngle; }
   int GetTextAngle() const { return (int)fTextAngle; }
};


private:
   std::string fText{};

   /// Text's X position
   double fX{0.};

   /// Text's Y position
   double fY{0.};

   /// Text attributes
   DrawingOpts fOpts;

public:
   TText() = default;

   TText(const std::string &str) : fText(str) {}
   TText(double x, double y, const std::string &str) : fText(str), fX(x), fY(y) {}

   void SetText(const std::string &txt) { fText = txt; }

   std::string GetText() const { return fText; }

   void SetPosition(double x, double y)
   {
      fX = x;
      fY = y;
   }

   double GetX() const { return fX; }

   double GetY() const { return fY; }

   /// Get the drawing options.
   DrawingOpts &GetOptions() { return fOpts; }
   const DrawingOpts &GetOptions() const { return fOpts; }

   void Paint(Internal::TVirtualCanvasPainter &canv) final
   {
      canv.AddDisplayItem(
         std::make_unique<ROOT::Experimental::TOrdinaryDisplayItem<ROOT::Experimental::TText>>(this));
   }
};

inline std::shared_ptr<ROOT::Experimental::TText>
GetDrawable(const std::shared_ptr<ROOT::Experimental::TText> &text)
{
   /// A TText is a TDrawable itself.
   return text;
}

} // namespace Experimental
} // namespace ROOT

#endif
