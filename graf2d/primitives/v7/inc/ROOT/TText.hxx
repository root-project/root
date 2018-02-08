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

#include <ROOT/TDrawingAttrs.hxx>

#include <ROOT/TDrawable.hxx>
#include <ROOT/TPad.hxx>
#include <ROOT/TVirtualCanvasPainter.hxx>

#include <ROOT/TDrawingAttrs.hxx>

#include <initializer_list>
#include <memory>
#include <string>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::TText
 A text.
 */

class TText {
private:
   std::string fText{};

   /// Text's X position
   double fX{0.};

   /// Text's Y position
   double fY{0.};

public:
   TText() = default;

   TText(const std::string &str) : fText(str) {}

   void SetText(const std::string &txt) { fText = txt; }

   std::string GetText() const { return fText; }

   void SetPosition(double x, double y)
   {
      fX = x;
      fY = y;
   }

   double GetX() const { return fX; }

   double GetY() const { return fY; }
};

class TextDrawingOpts {
   TDrawingAttrOrRef<int> fLineWidth{"Text.Line.Width", 3}; ///< The line width.
   TDrawingAttrOrRef<TColor> fLineColor{"Text.Line.Color", TColor::kBlack}; ///< The line color.
   TDrawingAttrOrRef<TColor> fFillColor{"Text.Fill.Color", TColor::kInvisible}; ///< The fill color.

public:
   /// The color of the line.
   void SetLineColor(const TColor &col) { fLineColor = col; }
   TColor &GetLineColor() { return fLineColor.Get(); }
   const TColor &GetLineColor() const { return fLineColor.Get(); }

   /// The width of the line.
   void SetLineWidth(int width) { fLineWidth = width; }
   int GetLineWidth() { return (int)fLineWidth; }

   /// The fill color
   void SetFillColor(const TColor &col) { fFillColor = col; }
   TColor &GetFillColor() { return fFillColor.Get(); }
   const TColor &GetFillColor() const { return fFillColor.Get(); }
};

class TTextDrawable : public TDrawable {
private:
   /// Text string to be drawn

   Internal::TUniWeakPtr<ROOT::Experimental::TText> fText{};

   /// Text attributes
   TextDrawingOpts fOpts;

public:
   TTextDrawable() = default;

   TTextDrawable(const std::shared_ptr<ROOT::Experimental::TText> &txt)
      : TDrawable(), fText(txt)
   {
   }

   TextDrawingOpts &GetOptions() { return fOpts; }
   const TextDrawingOpts &GetOptions() const { return fOpts; }

   void Paint(Internal::TVirtualCanvasPainter &canv) final
   {
      canv.AddDisplayItem(new ROOT::Experimental::TOrdinaryDisplayItem<ROOT::Experimental::TTextDrawable>(this));
   }
};

inline std::unique_ptr<ROOT::Experimental::TTextDrawable>
GetDrawable(const std::shared_ptr<ROOT::Experimental::TText> &text)
{
   return std::make_unique<ROOT::Experimental::TTextDrawable>(text);
}

} // namespace Experimental
} // namespace ROOT

#endif
