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
class THistBinRef {
public:
  using Coord_t = typename Detail::THistImplBase<DIMENSION, PRECISION>::Coord_t;
  using HistImpl_t = Detail::THistImplBase<DIMENSION, PRECISION>;

private:
  size_t fIndex = 0; ///< Bin index
  HistImpl_t& fHist; ///< The bin's histogram.
  std::array_view<PRECISION> fBinContent; ///< Histogram's bin content

public:
  /// Construct from a histogram.
  THistBinRef(HistImpl_t& hist):
    fHist(hist), fBinContent(hist.GetContent()) {}

  /// Construct from a histogram.
  THistBinRef(HistImpl_t& hist, size_t idx):
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

template <int DIMENSION, class PRECISION>
class THistBinPtr {
  THistBinRef<DIMENSION, PRECISION> fRef;
public:
  const THistBinRef<DIMENSION, PRECISION>& operator->() const noexcept {
    return fRef;
  }
};


/**
 \class THistBinIter
 Iterates over the bins of a THist or THistImpl.
 */

template <int DIMENSION, class PRECISION>
class THistBinIter:
  public Internal::TIndexIter<THistBinRef<DIMENSION, PRECISION>,
                              THistBinPtr<DIMENSION, PRECISION>> {
public:
  using HistImpl_t = Detail::THistImplBase<DIMENSION, PRECISION>;
  using Ref_t = THistBinRef<DIMENSION, PRECISION>;
  using Ptr_t = THistBinPtr<DIMENSION, PRECISION>;

private:
  size_t fIndex = 0; ///< Bin index
  HistImpl_t& fHist; ///< The histogram we iterate over.
  std::array_view<PRECISION> fBinContent; ///< Histogram's bin content

public:
  /// Construct a THistBinIter from a histogram.
  THistBinIter(HistImpl_t& hist):
    fHist(hist), fBinContent(fHist->GetBinContent()) {}

  /// Construct a THistBinIter from a histogram, setting the current index.
  THistBinIter(HistImpl_t& hist, size_t idx):
    fIndex(idx), fHist(hist), fBinContent(fHist->GetBinContent()) {}

  ///\{
  ///\name Value access
  Ref_t operator*() const noexcept {
    return Ref_t{fIndex, fHist, fBinContent};
  }
  Ptr_t operator->() const noexcept {
    return Ptr_t{{fIndex, fHist, fBinContent}};
  }
  ///\}
};

// FIXME: implement STATISTICS access!

} // namespace Experimental
} // namespace ROOT

#endif
