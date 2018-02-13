/// \file ROOT/THistDrawingOpts.h
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

#ifndef ROOT7_THistDrawingOpts
#define ROOT7_THistDrawingOpts

#include <ROOT/TDrawingAttr.hxx>
#include <ROOT/TDrawingOptsBase.hxx>
#include <ROOT/TStringEnumAttr.hxx>

namespace ROOT {
namespace Experimental {

template <int DIMENSION>
class THistDrawingOpts {
   static_assert(DIMENSION != 0, "Cannot draw 0-dimensional histograms!");
   static_assert(DIMENSION > 3, "Cannot draw histograms with more than 3 dimensions!");
   static_assert(DIMENSION < 3, "This should have been handled by the specializations below?!");
};

/** \class THistDrawingOpts<1>
 Drawing options for a 1D histogram.
 */
template <>
class THistDrawingOpts<1>: public TDrawingOptsBase {
public:
   enum class EStyle { kBar, kText };

private:
   static const TStringEnumAttrSet &Styles() {
      static const TStringEnumAttrSet styles{"hist", "bar", "text"};
      return styles;
   }
   TDrawingAttrOrRef<TStringEnumAttr> fStyle{*this, "Hist.1D.Style", 0, Styles()};
   TDrawingAttrOrRef<TColor> fLineColor{*this, "Hist.1D.Line.Color"};
   TDrawingAttrOrRef<int> fLineWidth{*this, "Hist.1D.Line.Width"};

public:
   EStyle GetStyle() const { return static_cast<EStyle>(fStyle.Get().GetIndex()); }
   void SetStyle(EStyle style) { fStyle.Get().SetIndex(static_cast<int>(style)); }

   TColor GetLineColor() const { return fLineColor.Get(); }
   void SetLineColor(const TColor& col) { fLineColor = col; }

   int GetLineWidth() const { return fLineWidth.Get(); }
   void SetLineWidth(int width) { fLineWidth = width; }
};

/** \class THistDrawingOpts<2>
 Drawing options for a 2D histogram.
 */
template <>
class THistDrawingOpts<2>: public TDrawingOptsBase {
public:
   enum class EStyle { kBox, kSurf, kText };

private:
   static const TStringEnumAttrSet &Styles() {
      static const TStringEnumAttrSet styles{"box", "surf", "text"};
      return styles;
   }
   TDrawingAttrOrRef<TStringEnumAttr> fStyle{*this, "Hist.2D.Style", 0, Styles()};
   TDrawingAttrOrRef<TColor> fLineColor{*this, "Hist.2D.Line.Color"};
   TDrawingAttrOrRef<int> fLineWidth{*this, "Hist.2D.Line.Width"};

public:
   EStyle GetStyle() const { return static_cast<EStyle>(fStyle.Get().GetIndex()); }
   void SetStyle(EStyle style) { fStyle.Get().SetIndex(static_cast<int>(style)); }

   TColor GetLineColor() const { return fLineColor.Get(); }
   void SetLineColor(const TColor& col) { fLineColor = col; }

   int GetLineWidth() const { return fLineWidth.Get(); }
   void SetLineWidth(int width) { fLineWidth = width; }
};

/** \class THistDrawingOpts<3>
 Drawing options for a 3D histogram.
 */
template <>
class THistDrawingOpts<3>: public TDrawingOptsBase {
public:
   enum class EStyle { kBox, kIso };

private:
   static const TStringEnumAttrSet &Styles() {
      static const TStringEnumAttrSet styles{"box", "iso"};
      return styles;
   }
   TDrawingAttrOrRef<TStringEnumAttr> fStyle{*this, "Hist.3D.Style", 0, Styles()};
   TDrawingAttrOrRef<TColor> fLineColor{*this, "Hist.3D.Line.Color"};
   TDrawingAttrOrRef<int> fLineWidth{*this, "Hist.3D.Line.Width"};

public:
   EStyle GetStyle() const { return static_cast<EStyle>(fStyle.Get().GetIndex()); }
   void SetStyle(EStyle style) { fStyle.Get().SetIndex(static_cast<int>(style)); }

   TColor GetLineColor() const { return fLineColor.Get(); }
   void SetLineColor(const TColor& col) { fLineColor = col; }

   int GetLineWidth() const { return fLineWidth.Get(); }
   void SetLineWidth(int width) { fLineWidth = width; }
};

} // namespace Experimental
} // namespace ROOT

#endif
