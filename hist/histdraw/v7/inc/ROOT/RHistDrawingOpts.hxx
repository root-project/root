/// \file ROOT/RHistDrawingOpts.h
/// \ingroup HistDraw ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-09-04
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RHistDrawingOpts
#define ROOT7_RHistDrawingOpts

#include <ROOT/RDrawingAttr.hxx>
#include <ROOT/RDrawingOptsBase.hxx>
#include <ROOT/RStringEnumAttr.hxx>

namespace ROOT {
namespace Experimental {

template <int DIMENSION>
class RHistDrawingOpts {
   static_assert(DIMENSION != 0, "Cannot draw 0-dimensional histograms!");
   static_assert(DIMENSION > 3, "Cannot draw histograms with more than 3 dimensions!");
   static_assert(DIMENSION < 3, "This should have been handled by the specializations below?!");
};

/** \class RHistDrawingOpts<1>
 Drawing options for a 1D histogram.
 */
template <>
class RHistDrawingOpts<1>: public RDrawingOptsBase {
public:
   enum class EStyle { kBar, kText };

private:
   static const RStringEnumAttrSet &Styles() {
      static const RStringEnumAttrSet styles{"hist", "bar", "text"};
      return styles;
   }
   RDrawingAttr<RStringEnumAttr<EStyle>> fStyle{*this, "Hist.1D.Style", EStyle::kBar, Styles()};
   RDrawingAttr<RColor> fLineColor{*this, "Hist.1D.Line.Color"};
   RDrawingAttr<int> fLineWidth{*this, "Hist.1D.Line.Width"};

public:
   EStyle GetStyle() const { return fStyle.Get().GetIndex(); }
   RDrawingAttr<RStringEnumAttr<EStyle>> &GetStyle() { return fStyle; }
   void SetStyle(EStyle style) { fStyle.Get().SetIndex(style); }

   RColor GetLineColor() const { return fLineColor.Get(); }
   RDrawingAttr<RColor> &GetLineColor() { return fLineColor; }
   void SetLineColor(const RColor& col) { fLineColor = col; }

   int GetLineWidth() const { return fLineWidth.Get(); }
   RDrawingAttr<int> &GetLineWidth() { return fLineWidth; }
   void SetLineWidth(int width) { fLineWidth = width; }
};

/** \class RHistDrawingOpts<2>
 Drawing options for a 2D histogram.
 */
template <>
class RHistDrawingOpts<2>: public RDrawingOptsBase {
public:
   enum class EStyle { kBox, kSurf, kText };

private:
   static const RStringEnumAttrSet &Styles() {
      static const RStringEnumAttrSet styles{"box", "surf", "text"};
      return styles;
   }
   RDrawingAttr<RStringEnumAttr<EStyle>> fStyle{*this, "Hist.2D.Style", EStyle::kBox, Styles()};
   RDrawingAttr<RColor> fLineColor{*this, "Hist.2D.Line.Color"};
   RDrawingAttr<int> fLineWidth{*this, "Hist.2D.Line.Width"};

public:
   EStyle GetStyle() const { return fStyle.Get().GetIndex(); }
   RDrawingAttr<RStringEnumAttr<EStyle>> &GetStyle() { return fStyle; }
   void SetStyle(EStyle style) { fStyle.Get().SetIndex(style); }

   RColor GetLineColor() const { return fLineColor.Get(); }
   RDrawingAttr<RColor> &GetLineColor() { return fLineColor; }
   void SetLineColor(const RColor& col) { fLineColor = col; }

   int GetLineWidth() const { return fLineWidth.Get(); }
   RDrawingAttr<int> &GetLineWidth() { return fLineWidth; }
   void SetLineWidth(int width) { fLineWidth = width; }
};

/** \class RHistDrawingOpts<3>
 Drawing options for a 3D histogram.
 */
template <>
class RHistDrawingOpts<3>: public RDrawingOptsBase {
public:
   enum class EStyle { kBox, kIso };

private:
   static const RStringEnumAttrSet &Styles() {
      static const RStringEnumAttrSet styles{"box", "iso"};
      return styles;
   }
   RDrawingAttr<RStringEnumAttr<EStyle>> fStyle{*this, "Hist.3D.Style", EStyle::kBox, Styles()};
   RDrawingAttr<RColor> fLineColor{*this, "Hist.3D.Line.Color"};
   RDrawingAttr<int> fLineWidth{*this, "Hist.3D.Line.Width"};

public:
   EStyle GetStyle() const { return fStyle.Get().GetIndex(); }
   RDrawingAttr<RStringEnumAttr<EStyle>> &GetStyle() { return fStyle; }
   void SetStyle(EStyle style) { fStyle.Get().SetIndex(style); }

   RColor GetLineColor() const { return fLineColor.Get(); }
   RDrawingAttr<RColor> &GetLineColor() { return fLineColor; }
   void SetLineColor(const RColor& col) { fLineColor = col; }

   int GetLineWidth() const { return fLineWidth.Get(); }
   RDrawingAttr<int> &GetLineWidth() { return fLineWidth; }
   void SetLineWidth(int width) { fLineWidth = width; }
};

} // namespace Experimental
} // namespace ROOT

#endif
