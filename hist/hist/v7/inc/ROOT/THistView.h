/// \file ROOT/THistView.h
/// \ingroup Hist
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-08-06

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_THistView
#define ROOT7_THistView

#include "ROOT/THist.h"

namespace ROOT {

/*
 * Need THist::iterator for full range, takes a predicate for "in range?"
 * Returns true for THist; for THistView, checks range, returns false if not in
 * range. i+= 7 then does i++ seven times and checks at each step.
 * iterator is simply an int with a predicate functor. end is end of the
 * histogram - i.e. the number of bins (incl over / underflow).
 *
 * Add is then an operation (through a functor) on two bins.
 *
 * Drawing: need adaptor from THist<n,p>::GetBinContent(...) to
 * THistPrecNormalizer<n>::Get(i) that casts the bin content to a double. That
 * should be in internal but outside the drawing library (that needs to
 * communicate through abstract interfaces and can thus not instantiate
 * templates with user precision parameters.
 */

template <class HISTVIEW>
struct THistViewOutOfRange {
  HISTVIEW& fHistView;
  bool operator()(int idx) { return fHistView.IsBinOutOfRange(idx); }
};

/**
 \class THistView
 A view on a histogram, selecting a range on a subset of dimensions.
 */
template <int DIMENSIONS, class PRECISION>
class THistView {
public:
  using Hist_t = THist<DIMENSIONS, PRECISION>;
  using AxisIter_t = Hist::AxisIter_t<DIMENSIONS>;
  using AxisRange_t = Hist::AxisIterRange_t<DIMENSIONS>;

  using const_iterator = Internal::THistBinIter<THistViewOutOfRange>;

  THistView(Hist_t& hist, int nbins, const AxisRange_t& range):
     fHist(hist), fNBins(nbins), fRange(range) {}

  bool IsBinOutOfRange(int idx) const noexcept {
    // TODO: implement IsBinOutOfRange()!
  }

  void SetRange(int axis, double from, double to) {
    TAxisView axis = fHist.GetImpl()->GetAxis(axis);
    fRange[axis] = axis.FindBin(from);
    fRange[axis] = axis.FindBin(to);
  }

  const_iterator begin() const noexcept {
    int beginidx = 0;
    size_t nbins = fHist.GetNBins();
    while (IsBinOutOfRange(beginidx) && beginidx < nbins)
      ++beginidx;
    return const_iterator(beginidx, THistViewOutOfRange(*this));
  }

  const_iterator end() const noexcept  {
    return const_iterator(fHist.GetImpl().GetNBins(),
                          Internal::HistIterFullRange_t());
  }

private:
  Hist_t& fHist;
  int fNBins = 0;
  AxisRange_t fRange;
};




} // namespace ROOT

#endif
