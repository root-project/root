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

#include <ROOT/TDrawingOptsBase.hxx>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::THistDrawingOptsBase

Stores drawing options for a histogram. This class contains the properties
that are independent of the histogram dimensionality.

Customize the defaults with `THistDrawingOpts<1>::SetLineColor(TColor::kRed);` or by modifying the
style configuration file `rootstylerc` (e.g. $ROOTSYS/etc/system.rootstylerc or ~/.rootstylerc).
 */
template <class DERIVED>
class THistDrawingOptsBase: public TDrawingOptsBase<DERIVED> {
   /// Index of the line color in TCanvas's color table.
   //TOptsAttrRef<TColor> fLineColorIndex{*this, "Hist.Line.Color"};
   //TOptsAttrRef<long long> fLineColorIndex{*this, "Hist.Line.Width"};
   TLineDrawingOpts fLine{*this, "Hist", {kBlack, 3}};
   
public:
   THistDrawingOptsBase() = default;
   THistDrawingOptsBase(TPadBase &pad): TDrawingOptsBase<DERIVED>(pad) {}

   /// The color of the histogram line.
   void SetLineColor(const TColor &col) { this->Update(fLineColorIndex, col); }
   TColor &GetLineColor() { return this->Get(fLineColorIndex); }
   const TColor &GetLineColor() const { return this->Get(fLineColorIndex); }
};

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
class THistDrawingOpts<1>: public THistDrawingOptsBase<THistDrawingOpts<1>> {
   enum class EStyle { kErrors, kBar, kText };
   TOptsAttrRef<long long> fStyle{*this, "Hist.1D.Style"};

public:
   THistDrawingOpts() = default;
   THistDrawingOpts(TPadBase &pad): THistDrawingOptsBase<THistDrawingOpts<1>>(pad) {}
};

/** \class THistDrawingOpts<2>
 Drawing options for a 1D histogram.
 */
template <>
class THistDrawingOpts<2>: public THistDrawingOptsBase<THistDrawingOpts<1>> {
   enum class EStyle { kErrors, kBar, kText };
   TOptsAttrRef<long long> fStyle{*this, "Hist.2D.Style"};

public:
   THistDrawingOpts() = default;
   THistDrawingOpts(TPadBase &pad): THistDrawingOptsBase<THistDrawingOpts<1>>(pad) {}
};

/** \class THistDrawingOpts<3>
 Drawing options for a 1D histogram.
 */
template <>
class THistDrawingOpts<3>: public THistDrawingOptsBase<THistDrawingOpts<1>> {
   enum class EStyle { kErrors, kBar, kText };
   TOptsAttrRef<long long> fStyle{*this, "Hist.3D.Style"};

public:
   THistDrawingOpts() = default;
   THistDrawingOpts(TPadBase &pad): THistDrawingOptsBase<THistDrawingOpts<1>>(pad) {}
};

} // namespace Experimental
} // namespace ROOT

#endif
