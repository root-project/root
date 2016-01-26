/// \file ROOT/THistBinIter.h
/// \ingroup Hist ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \ingroup Hist ROOT7
/// \date 2015-08-07
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_THistBinIter
#define ROOT7_THistBinIter

#include <ROOT/TIndexIter.h>
#include <ROOT/THistImpl.h>

namespace ROOT {
namespace Experimental {

template <int DIMENSION, class PRECISION> class THistBinIter;

/**
 \class THistBin
 Represents a bin. Value of the bin iteration.
 */

template <int DIMENSION, class PRECISION>
class THistBin {
  size_t fIndex = 0; ///< Bin index
  Detail::THistImplBase<DIMENSION, PRECISION>& fHist; ///< The bin's histogram.
  std::array_view<PRECISION> fBinContent; ///< Histogram's bin content

public:
  using Coord_t = typename Detail::THistImplBase<DIMENSION, PRECISION>::Coord_t;

  /// Construct from a histogram.
  THistBin(Detail::THistImplBase<DIMENSION, PRECISION>& hist):
    fHist(hist), fBinContent(hist.GetContent()) {}

  /// Construct from a histogram.
  THistBin(Detail::THistImplBase<DIMENSION, PRECISION>& hist, size_t idx):
    fIndex(idx), fHist(hist), fBinContent(hist.GetContent()) {}

  /// Get the bin content.
  PRECISION Get() const { return fBinContent[fIndex]; }

  /// Get the bin center as an array over all dimensions.
  Coord_t GetCenter() const { return fHist.GetBinCenter(fIndex); }

  /// Get the bin lower edge as an array over all dimensions.
  Coord_t GetFrom() const { return fHist.GetBinFrom(fIndex); }

  /// Get the bin upper edge as an array over all dimensions.
  Coord_t GetTo() const { return fHist.GetBinTo(fIndex); }

  friend class THistBinIter<DIMENSION, PRECISION>;
};



/**
 \class THistBinIter
 Iterates over the bins of a THist or THistImpl.
 */

template <int DIMENSION, class PRECISION>
class THistBinIter:
  public Internal::TIndexIter<THistBin<DIMENSION, PRECISION>,
    THistBin<DIMENSION, PRECISION>> {
public:
  using Value_t = THistBin<DIMENSION, PRECISION>;

private:
  Value_t fCurrentBin; ///< Current iteration's bin

  /// Get the current index.
  size_t& GetIndex() noexcept { return fCurrentBin.fIndex; }
  /// Get the current index.
  size_t GetIndex() const noexcept { return fCurrentBin.fIndex; }

public:
  /// Construct a THistBinIter from a histogram.
  THistBinIter(Detail::THistImplBase<DIMENSION, PRECISION>& hist):
    fCurrentBin(hist) {}

  /// Construct a THistBinIter from a histogram, setting the current index.
  THistBinIter(Detail::THistImplBase<DIMENSION, PRECISION>& hist, size_t idx):
    fCurrentBin(hist, idx) {}

  ///\{
  ///\name Value access
  Value_t& operator*() const noexcept { return fCurrentBin; }
  Value_t* operator->() const noexcept { return &fCurrentBin; }
  ///\}
};

// FIXME: implement STATISTICS access!

} // namespace Experimental
} // namespace ROOT

#endif
