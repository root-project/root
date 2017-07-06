/// \file ROOT/THistDrawOptions.h
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

#ifndef ROOT7_THistDrawOptions
#define ROOT7_THistDrawOptions

namespace ROOT {
namespace Experimental {

namespace Internal {

template <int DIMENSION>
struct THistDrawOptionsEnum;

/// Specialization containing 1D hist drawing options.
template <>
struct THistDrawOptionsEnum<1> {
   enum EOpts { kErrors, kBar, kText };
};

/// Specialization containing 2D hist drawing options.
template <>
struct THistDrawOptionsEnum<2> {
   enum EOpts { kBox, kText, kLego };
};

/// Specialization containing 3D hist drawing options.
template <>
struct THistDrawOptionsEnum<3> {
   enum EOpts { kLego, kIso };
};

} // namespace Internal

/** \class THistDrawOptions
 Drawing options for a histogram with DIMENSIONS
 */
template <int DIMENSION>
class THistDrawOptions {
   int fOpts;

public:
   THistDrawOptions() = default;
   constexpr THistDrawOptions(typename Internal::THistDrawOptionsEnum<DIMENSION>::EOpts opt): fOpts(2 >> opt) {}
};

namespace Hist {
static constexpr const THistDrawOptions<2> box(Internal::THistDrawOptionsEnum<2>::kBox);
static constexpr const THistDrawOptions<2> text(Internal::THistDrawOptionsEnum<2>::kText);
} // namespace Hist

} // namespace Experimental
} // namespace ROOT

#endif
