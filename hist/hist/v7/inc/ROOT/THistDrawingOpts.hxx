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

namespace Internal {

template <int DIMENSION>
class THistDrawingOptsBase: public TDrawingOptsBase {
   /// Index of the line color in TCanvas's color table.
   size_t fLineColorIndex = (size_t) -1;
   /// The pad containing this Drawable, used to register colors.
   TPadBase& fPad;

public:
   THistDrawingOptsBase(TPadBase& pad) = default;

   void SetLineColor(const TColor& col) {
       fPad.RegisterColor(fLineColorIndex, col);
   }
};

/// Specialization containing 1D hist drawing options.
template <>
struct THistDrawingOptsEnum<1> {
   enum EOpts { kErrors, kBar, kText };
};

/// Specialization containing 2D hist drawing options.
template <>
struct THistDrawingOptsEnum<2> {
   enum EOpts { kBox, kText, kLego };
};

/// Specialization containing 3D hist drawing options.
template <>
struct THistDrawingOptsEnum<3> {
   enum EOpts { kLego, kIso };
};

} // namespace Internal

/** \class THistDrawingOpts
 Drawing options for a histogram with DIMENSIONS
 */
template <int DIMENSION>
class THistDrawingOpts {
   int fOpts;

public:
   THistDrawingOpts() = default;
   constexpr THistDrawingOpts(typename Internal::THistDrawingOptsEnum<DIMENSION>::EOpts opt): fOpts(2 >> opt) {}
};

namespace Hist {
static constexpr const THistDrawingOpts<2> box(Internal::THistDrawingOptsEnum<2>::kBox);
static constexpr const THistDrawingOpts<2> text(Internal::THistDrawingOptsEnum<2>::kText);
} // namespace Hist

} // namespace Experimental
} // namespace ROOT

#endif
