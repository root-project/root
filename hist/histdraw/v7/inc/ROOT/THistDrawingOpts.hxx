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

#include <ROOT/TDrawingAttrs.hxx>
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
class THistDrawingOpts<1> {
   enum class EStyle { kBar, kText };
   static const TStringEnumAttrSet fgStyles;
   TDrawingAttrOrRef<TStringEnumAttr> fStyle{"Hist.1D.Style", 0, fgStyles};
   TDrawingAttrOrRef<TColor> fLineColor{"Hist.1D.Line.Color"};
   TDrawingAttrOrRef<int> fLineWidth{"Hist.1D.Line.Width"};
public:
   THistDrawingOpts() = default;
};

const TStringEnumAttrSet THistDrawingOpts<1>::fgStyles{"hist", "bar", "text"};


/** \class THistDrawingOpts<2>
 Drawing options for a 1D histogram.
 */
template <>
class THistDrawingOpts<2> {
   enum class EStyle { kBox, kSurf, kText };
   static const TStringEnumAttrSet fgStyles;
   TDrawingAttrOrRef<TStringEnumAttr> fStyle{"Hist.2D.Style", 0, fgStyles};
   TDrawingAttrOrRef<TColor> fLineColor{"Hist.2D.Line.Color"};
   TDrawingAttrOrRef<int> fLineWidth{"Hist.2D.Line.Width"};
public:
   THistDrawingOpts() = default;
};

const TStringEnumAttrSet THistDrawingOpts<2>::fgStyles{"box", "surf", "text"};


/** \class THistDrawingOpts<3>
 Drawing options for a 1D histogram.
 */
template <>
class THistDrawingOpts<3> {
   enum class EStyle { kBox, kIso };
   static const TStringEnumAttrSet fgStyles;
   TDrawingAttrOrRef<TStringEnumAttr> fStyle{"Hist.3D.Style", 0, fgStyles};
   TDrawingAttrOrRef<TColor> fLineColor{"Hist.3D.Line.Color"};
   TDrawingAttrOrRef<int> fLineWidth{"Hist.3D.Line.Width"};
public:
   THistDrawingOpts() = default;
};

const TStringEnumAttrSet THistDrawingOpts<3>::fgStyles{"box", "iso"};

} // namespace Experimental
} // namespace ROOT

#endif
