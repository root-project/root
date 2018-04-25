/// \file ROOT/TLine.hxx
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

#ifndef ROOT7_TLine
#define ROOT7_TLine

#include <ROOT/TDrawable.hxx>
#include <ROOT/TDrawingAttr.hxx>
#include <ROOT/TDrawingOptsBase.hxx>
#include <ROOT/TPadPos.hxx>
#include <ROOT/TPad.hxx>
#include <ROOT/TPadPainter.hxx>

#include <initializer_list>
#include <memory>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::TLine
 A simple line.
 */

class TLine : public TDrawableBase<TLine> {
public:

/** class ROOT::Experimental::TLine::DrawingOpts
 Drawing options for TLine.
 */

class DrawingOpts: public TDrawingOptsBase {
   TDrawingAttr<TColor> fLineColor{*this, "Line.Color", TColor::kBlack};   ///< The line color.
   TDrawingAttr<int> fLineWidth{*this, "Line.Width", 1. };                 ///< The line width.
   TDrawingAttr<int> fLineStyle{*this, "Line.Style", 1.};                  ///< The line style.
   TDrawingAttr<float>  fLineOpacity{*this, "Line.Opacity", 1.};           ///< The line opacity.

public:
   /// The color of the line.
   void SetLineColor(const TColor &col) { fLineColor = col; }
   TDrawingAttr<TColor> &GetLineColor() { return fLineColor; }
   const TColor &GetLineColor() const   { return fLineColor.Get(); }

   ///The width of the line.
    void SetLineWidth(int width) { fLineWidth = width; }
    TDrawingAttr<int> &GetLineWidth() { return fLineWidth; }
    int GetLineWidth() const   { return (int)fLineWidth; }

    ///The style of the line.
    void SetLineStyle(int style) { fLineStyle = style; }
    TDrawingAttr<int> &GetLineStyle() { return fLineStyle; }
    int GetLineStyle() const { return (int)fLineStyle; }

    ///The opacity of the line.
    void SetLineColorAlpha(float opacity) { fLineOpacity = opacity; }
    TDrawingAttr<float> &GetLineColorAlpha() { return fLineOpacity; }
    float GetLineColorAlpha() const { return (float)fLineOpacity; }
};


private:

   /// Line's coordinates

   TPadPos fP1;           ///< 1st point
   TPadPos fP2;           ///< 2nd point

   /// Line attributes
   DrawingOpts fOpts;

public:
   TLine() = default;

   TLine(const TPadPos& p1, const TPadPos& p2) : fP1(p1), fP2(p2) {}

   void SetP1(const TPadPos& p1) { fP1 = p1; }
   void SetP2(const TPadPos& p2) { fP2 = p2; }

   const TPadPos& GetP1() const { return fP1; }
   const TPadPos& GetP2() const { return fP2; }

   /// Get the drawing options.
   DrawingOpts &GetOptions() { return fOpts; }
   const DrawingOpts &GetOptions() const { return fOpts; }

   void Paint(Internal::TPadPainter &topPad) final
   {
      topPad.AddDisplayItem(
         std::make_unique<ROOT::Experimental::TOrdinaryDisplayItem<ROOT::Experimental::TLine>>(this));
   }
};

inline std::shared_ptr<ROOT::Experimental::TLine>
GetDrawable(const std::shared_ptr<ROOT::Experimental::TLine> &line)
{
   /// A TLine is a TDrawable itself.
   return line;
}

} // namespace Experimental
} // namespace ROOT

#endif
