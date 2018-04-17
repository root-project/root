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

   double fX1{0.};           ///< X of 1st point
   double fY1{0.};           ///< Y of 1st point
   double fX2{0.};           ///< X of 2nd point
   double fY2{0.};           ///< Y of 2nd point

   /// Line attributes
   DrawingOpts fOpts;

public:
   TLine() = default;

   TLine(double x1, double y1, double x2, double y2 ) : fX1(x1), fY1(y1), fX2(x2), fY2(y2) {}

   void SetX1(double x1) { fX1 = x1; }
   void SetY1(double y1) { fY1 = y1; }
   void SetX2(double x2) { fX2 = x2; }
   void SetY2(double y2) { fY2 = y2; }

   double GetX1() const { return fX1; }
   double GetY1() const { return fY1; }
   double GetX2() const { return fX2; }
   double GetY2() const { return fY2; }

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
