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

/** \class ROOT::Experimental::THistCoreAttrs

Stores drawing attributes for a histogram. This class contains the properties
that are independent of the histogram dimensionality, and used by at least one of
the visualizations (scatter, bar, etc).

Customize the defaults with `THistDrawingOpts<1>::Default().SetLineColor(TColor::kRed);` or by modifying the
style configuration file `rootstylerc` (e.g. $ROOTSYS/etc/system.rootstylerc or ~/.rootstylerc).
 */
struct THistCoreAttrs {
   /// Index of the line color in TCanvas's color table.
   //TDrawingAttrRef<TColor> fLineColorIndex{*this, "Hist.Line.Color"};
   //TDrawingAttrRef<long long> fLineColorIndex{*this, "Hist.Line.Width"};
   TLineAttrs fLine; ///< The histogram line attributes
   TFillAttrs fFill; ///< The histogram fill attributes

   THistCoreAttrs() = default;
   THistCoreAttrs(TDrawingOptsBaseNoDefault &opts, const std::string &name):
      fLine{opts, name + ".Line", TColor::kBlack, TLineAttrs::Width{3}},
      fFill{opts, name + ".Fill", TColor::kWhite}
      {}
};

template <int DIMENSION>
class THistDrawingOpts {
   static_assert(DIMENSION != 0, "Cannot draw 0-dimensional histograms!");
   static_assert(DIMENSION > 3, "Cannot draw histograms with more than 3 dimensions!");
   static_assert(DIMENSION < 3, "This should have been handled by the specializations below?!");
};

/** \class THistDrawingOptsBase
 Core ingredients (that do not depend on the dimensionality) or histogram drawing options.
 */

template <class DERIVED>
class THistDrawingOptsBase: public TDrawingOptsBase<DERIVED> {
   THistCoreAttrs fHistAttrs{*this, "Hist"}; ///< Basic histogram attributes (line, fill etc)

protected:
   THistDrawingOptsBase() = default;
   THistDrawingOptsBase(TPadBase &pad, const std::string &name):
      TDrawingOptsBase<DERIVED>(pad, name + ".Hist") {}

public:
   /// The color of the histogram line.
   void SetLineColor(const TColor &col) { this->Update(fHistAttrs.fLine.fColor, col); }
   TColor &GetLineColor() { return this->Get(fHistAttrs.fLine.fColor); }
   const TColor &GetLineColor() const { return this->Get(fHistAttrs.fLine.fColor); }

   /// The width of the histogram line.
   void SetLineWidth(TLineAttrs::Width width) { this->Update(fHistAttrs.fLine.fWidth, width); }
   TLineAttrs::Width &GetLineWidth() { return this->Get(fHistAttrs.fLine.fWidth); }
   const TLineAttrs::Width GetLineWidth() const { return this->Get(fHistAttrs.fLine.fWidth); }

   /// The color of the histogram line.
   void SetFillColor(const TColor &col) { this->Update(fHistAttrs.fFill.fColor, col); }
   TColor &GetFillColor() { return this->Get(fHistAttrs.fFill.fColor); }
   const TColor &GetFillColor() const { return this->Get(fHistAttrs.fFill.fColor); }
};

/** \class THistDrawingOpts<1>
 Drawing options for a 1D histogram.
 */
template <>
class THistDrawingOpts<1>: public THistDrawingOptsBase<THistDrawingOpts<1>> {
   enum class EStyle { kBar, kText };
   TDrawingAttrRef<long long> fStyle{*this, "Style", 0, {"hist", "bar", "text"}};

public:
   THistDrawingOpts() = default;
   explicit THistDrawingOpts(TPadBase &pad): THistDrawingOptsBase<THistDrawingOpts<1>>(pad, "1D") {}

};

/** \class THistDrawingOpts<2>
 Drawing options for a 1D histogram.
 */
template <>
class THistDrawingOpts<2>: public THistDrawingOptsBase<THistDrawingOpts<2>> {
   enum class EStyle { kBox, kSurf, kText };
   TDrawingAttrRef<long long> fStyle{*this, "Style", 0, {"box", "surf", "text"}};

public:
   THistDrawingOpts() = default;
   explicit THistDrawingOpts(TPadBase &pad): THistDrawingOptsBase<THistDrawingOpts<2>>(pad, "2D") {}
};

/** \class THistDrawingOpts<3>
 Drawing options for a 1D histogram.
 */
template <>
class THistDrawingOpts<3>: public THistDrawingOptsBase<THistDrawingOpts<3>> {
   enum class EStyle { kBox, kIso };
   TDrawingAttrRef<long long> fStyle{*this, "Style", 0, {"box", "iso"}};

public:
   THistDrawingOpts() = default;
   explicit THistDrawingOpts(TPadBase &pad): THistDrawingOptsBase<THistDrawingOpts<3>>(pad, "3D") {}
};

} // namespace Experimental
} // namespace ROOT

#endif
