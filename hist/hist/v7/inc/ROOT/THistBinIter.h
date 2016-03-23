/// \file ROOT/THistBinIter.h
/// \ingroup Hist ROOT7
/// \author Axel Naumann <axel@cern.ch>
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

namespace ROOT {
namespace Experimental {
namespace Detail {

/**
\class THistBinRef
Represents a bin reference. Value of the bin iteration.

Provides access to bin content, bin geometry (from, to, center), and statistics
(for instance higher moments) associated to the bin.
*/

template <class HISTIMPL>
class THistBinRef {
public:
  using HistImpl_t = HISTIMPL;
  using Coord_t = typename HISTIMPL::Coord_t;
  using Weight_t = typename HISTIMPL::Weight_t;

private:
  size_t fIndex = 0; ///< Bin index
  HistImpl_t* fHist; ///< The bin's histogram.

public:
  /// Construct from a histogram.
  THistBinRef(HistImpl_t& hist, size_t idx): fIndex(idx), fHist(&hist) {}

  /// \{
  /// \name Statistics operations
  /// Get the bin content (or reference to it, for non-const HistImpl_t).
  auto GetContent() { return fHist->GetBinContent(fIndex); }

  /// Get the bin uncertainty.
  auto GetUncertainty() const { return GetStat().GetUncertainty(); }

  /// Get a (const, for const HistImpl_t) reference to the bin-view of the
  /// histogram statistics (uncertainty etc).
  auto GetStat() const { return fHist->GetStat().GetView(fIndex); }
  /// \}

  /// \{
  /// \name Bin operations
  /// Get the bin center as an array over all dimensions.
  Coord_t GetBinCenter() const { return fHist->GetBinCenter(fIndex); }

  /// Get the bin lower edge as an array over all dimensions.
  Coord_t  GetBinFrom() const { return fHist->GetBinFrom(fIndex); }

  /// Get the bin upper edge as an array over all dimensions.
  Coord_t  GetBinTo() const { return fHist->GetBinTo(fIndex); }
  /// \}
};


/**
 \class THistBinPtr
 Points to a histogram bin (or actually a `THistBinRef`).
 */
template<class HISTIMPL>
class THistBinPtr {
public:
  using Ref_t = THistBinRef<HISTIMPL>;

  const Ref_t &operator->() const noexcept {
    return fRef;
  }

private:
  Ref_t fRef; ///< Underlying bin reference
};


/**
 \class THistBinIter
 Iterates over the bins of a THist or THistImpl.
 */

template<class HISTIMPL>
class THistBinIter:
  public Internal::TIndexIter<THistBinRef < HISTIMPL>, THistBinPtr<HISTIMPL>> {
public:
  using Ref_t = THistBinRef<HISTIMPL>;
  using Ptr_t = THistBinPtr<HISTIMPL>;

private:
  using IndexIter_t = Internal::TIndexIter <THistBinRef<HISTIMPL>, THistBinPtr<HISTIMPL>>;

  HISTIMPL &fHist; ///< The histogram we iterate over.

public:
  /// Construct a THistBinIter from a histogram.
  THistBinIter(HISTIMPL &hist):
    IndexIter_t(0), fHist(hist) { }

  /// Construct a THistBinIter from a histogram, setting the current index.
  THistBinIter(HISTIMPL &hist, size_t idx):
    IndexIter_t(idx), fHist(hist) { }

  ///\{
  ///\name Value access
  Ref_t operator*() const noexcept {
    return Ref_t{fHist, IndexIter_t::GetIndex()};
  }

  Ptr_t operator->() const noexcept {
    return Ptr_t{*this};
  }
  ///\}
};

} // namespace Detail
} // namespace Experimental
} // namespace ROOT

#endif
