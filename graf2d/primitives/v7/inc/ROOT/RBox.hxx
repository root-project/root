/// \file ROOT/RBox.hxx
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

#ifndef ROOT7_RBox
#define ROOT7_RBox

#include <ROOT/RDrawable.hxx>
#include <ROOT/RDrawingAttr.hxx>
#include <ROOT/RDrawingOptsBase.hxx>
#include <ROOT/RPadPos.hxx>
#include <ROOT/RPadPainter.hxx>

#include <initializer_list>
#include <memory>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::RBox
 A simple box.
 */

class RBox : public RDrawableBase<RBox> {
public:

/** class ROOT::Experimental::RBox::DrawingOpts
 Drawing options for RBox.
 */

class DrawingOpts: public RDrawingOptsBase {
   RDrawingAttr<RColor> fLineColor  {*this, "Line.Color"  , RColor::kBlack}; ///< The box line color.
   RDrawingAttr<int>    fLineWidth  {*this, "Line.Width"  , 1};              ///< The box line width.
   RDrawingAttr<int>    fLineStyle  {*this, "Line.Style"  , 1};              ///< The box line style.
   RDrawingAttr<float>  fLineOpacity{*this, "Line.Opacity", 1.};             ///< The box line opacity.
   RDrawingAttr<RColor> fFillColor  {*this, "Fill.Color"  , RColor::kBlack}; ///< The box fill color.
   RDrawingAttr<int>    fFillStyle  {*this, "Fill.Style"  , 1};              ///< The box line style.
   RDrawingAttr<float>  fFillOpacity{*this, "Fill.Opacity", 1.};             ///< The box fill opacity.
   RDrawingAttr<int>    fRoundWidth {*this, "Round.Width" , 0};              ///< Determines how wide the corners'rounding is.
   RDrawingAttr<int>    fRoundHeight{*this, "Round.Height", 0};              ///< Determines how high the corners'rounding is.

public:
   /// The color of the box line.
   void SetLineColor(const RColor &col) { fLineColor = col; }
   RDrawingAttr<RColor> &GetLineColor() { return fLineColor; }
   const RColor &GetLineColor() const   { return fLineColor.Get(); }

   /// The width of the box line.
   void SetLineWidth(int width) { fLineWidth = width; }
   RDrawingAttr<int> &GetLineWidth() { return fLineWidth; }
   int GetLineWidth() const   { return (int)fLineWidth; }

   /// The style of the box line.
   void SetLineStyle(int style) { fLineStyle = style; }
   RDrawingAttr<int> &GetLineStyle() { return fLineStyle; }
   int GetLineStyle() const { return (int)fLineStyle; }

   /// The opacity of the box line.
   void SetLineColorAlpha(float opacity) { fLineOpacity = opacity; }
   RDrawingAttr<float> &GetLineColorAlpha() { return fLineOpacity; }
   float GetLineColorAlpha() const { return (float)fLineOpacity; }

   /// The color of the box fill.
   void SetFillColor(const RColor &col) { fFillColor = col; }
   RDrawingAttr<RColor> &GetFillColor() { return fFillColor; }
   const RColor &GetFillColor() const   { return fFillColor.Get(); }

   /// The style of the box fill.
   void SetFillStyle(int style) { fFillStyle = style; }
   RDrawingAttr<int> &GetFillStyle() { return fFillStyle; }
   int GetFillStyle() const { return (int)fFillStyle; }

   /// How wide the corners'rounding is.
   void SetRoundWidth(int width) { fRoundWidth = width; }
   RDrawingAttr<int> &GetRoundWidth() { return fRoundWidth; }
   int GetRoundWidth() const { return (int)fRoundWidth; }

   /// How high the corners'rounding is.
   void SetRoundHeight(int height) { fRoundHeight = height; }
   RDrawingAttr<int> &GetRoundHeight() { return fRoundHeight; }
   int GetRoundHeight() const { return (int)fRoundHeight; }

   /// The opacity of the box fill.
   void SetFillColorAlpha(float opacity) { fFillOpacity = opacity; }
   RDrawingAttr<float> &GetFillColorAlpha() { return fFillOpacity; }
   float GetFillColorAlpha() const { return (float)fFillOpacity; }
};


private:

   /// Box's coordinates

   RPadPos fP1;           ///< 1st point, bottom left
   RPadPos fP2;           ///< 2nd point, top right

   /// Box's attributes
   DrawingOpts fOpts;

public:

   RBox() = default;

   RBox(const RPadPos& p1, const RPadPos& p2) : fP1(p1), fP2(p2) {}

   void SetP1(const RPadPos& p1) { fP1 = p1; }
   void SetP2(const RPadPos& p2) { fP2 = p2; }

   const RPadPos& GetP1() const { return fP1; }
   const RPadPos& GetP2() const { return fP2; }

   /// Get the drawing options.
   DrawingOpts &GetOptions() { return fOpts; }
   const DrawingOpts &GetOptions() const { return fOpts; }

   void Paint(Internal::RPadPainter &topPad) final
   {
      topPad.AddDisplayItem(
         std::make_unique<ROOT::Experimental::ROrdinaryDisplayItem<ROOT::Experimental::RBox>>(this));
   }
};

inline std::shared_ptr<ROOT::Experimental::RBox>
GetDrawable(const std::shared_ptr<ROOT::Experimental::RBox> &box)
{
   /// A RBox is a RDrawable itself.
   return box;
}

} // namespace Experimental
} // namespace ROOT

#endif
