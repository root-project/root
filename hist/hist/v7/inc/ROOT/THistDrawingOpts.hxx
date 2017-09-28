/// \file ROOT/THistDrawingOpts.h
/// \ingroup Hist ROOT7
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
class THistDrawingOptsBase: public TDrawingOptsBaseT<DERIVED> {
   /// Index of the line color in TCanvas's color table.
   TOptsAttrIdx<long long> fLineColorIndex;
public:
   using Attrs = typename TDrawingOptsBaseT<DERIVED>::Attrs;
   THistDrawingOptsBase(TPadBase &pad, const Attrs& attrs)
      : TDrawingOptsBaseT<DERIVED>(pad, Attrs(attrs).Add({fLineColorIndex, this->Default().fLineColorIndex})) {}

   /// The color of the histogram line.
   void SetLineColor(const TColor &col) { Update(fLineColorIndex, col); }
};


/** \class THistDrawingOpts
 Drawing options for a histogram with DIMENSIONS
 */
template <int DIMENSION>
class THistDrawingOptsBase: public THistDrawingOptsBase<THistDrawingOpts<DIMENSION>> {
   TOptsAttrIdx<long long> fStyle;
public:
   using Attrs = typename TDrawingOptsBaseT<DERIVED>::Attrs;
   THistDrawingOptsBase(TPadBase &pad, const Attrs& attrs)
      : THistDrawingOptsBase<THistDrawingOpts<DIMENSION>>(pad, Attrs(attrs).Add({fStyle, this->Default().fStyle})) {}
};

class THist1DDrawingOpts: public THistDrawingOptsBase<3> {
   enum class EStyle {
      kErrors,
      kBar,
      kText
   };
public:
};

} // namespace Experimental
} // namespace ROOT

#endif
