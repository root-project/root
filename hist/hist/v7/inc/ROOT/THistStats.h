/// \file ROOT/THistStats.h
/// \ingroup Hist ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-06-14
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_THistStats_h
#define ROOT7_THistStats_h

#include <vector>
#include "ROOT/RArrayView.h"

namespace ROOT {
namespace Experimental {

namespace Detail {
template <int DIMENSIONS, class PRECISION, class STATISTICS> class THistImplBase;
}

template<int DIMENSIONS, class PRECISION>
class THistStatEntries {
public:
  using Coord_t = std::array<double, DIMENSIONS>;
  using HistImpl_t = Detail::THistImplBase<DIMENSIONS, PRECISION, THistStatEntries>;
  using Weight_t = PRECISION;
  using BinStat_t = void;

  void Fill(const Coord_t& /*x*/, int /*binidx*/, Weight_t /*weightN*/ = 1.) {
    ++fEntries;
  }

  int64_t GetEntries() const { return fEntries; };

  double GetBinUncertainty(int binidx, const HistImpl_t& hist) const {
    return std::sqrt(std::fabs(hist.GetBinContent(binidx)));
  }
private:
  int64_t fEntries = 0;
};

template<int DIMENSIONS, class PRECISION>
class THistStatUncertainty: public THistStatEntries<DIMENSIONS, PRECISION> {
  std::vector<double> fSumWeightsSquared; ///< Sum of squared weights

public:
  /**
   View on a THistStatUncertainty for a given bin.
   */
  template <class HISTIMPL>
  class TBinStat {
  public:
    using HistImpl_t = HISTIMPL;
    auto GetSumWeightsSquared() const {
      return fHist.getStat().GetSumWeightsSquared()[fBinIndex];
    }

  private:
    HistImpl_t& fHist;
    size_t fBinIndex;
  };

  using Base_t = THistStatEntries<DIMENSIONS, PRECISION>;
  using typename Base_t::Coord_t;
  using typename Base_t::Weight_t;
  template <class HISTIMPL> using BinStat_t = TBinStat<HISTIMPL>;

  void Fill(const Coord_t &x, int binidx, Weight_t weight = 1.) {
    Base_t::Fill(x, binidx, weight);
    fSumWeightsSquared[binidx] += weight * weight;
  }

  const std::vector<double>& GetSumWeightsSquared() const { return fSumWeightsSquared; }
  std::vector<double>& GetSumWeightsSquared() { return fSumWeightsSquared; }

  double GetBinUncertainty(int binidx, const Detail::THistImplBase<DIMENSIONS,
    PRECISION, THistStatUncertainty>& /*hist*/) const {
    return std::sqrt(fSumWeightsSquared[binidx]);
  }
};

template<int DIMENSIONS, class PRECISION>
class THistStatMomentUncert: public THistStatUncertainty<DIMENSIONS, PRECISION> {
public:
  using Base_t = THistStatUncertainty<DIMENSIONS, PRECISION>;
  using typename Base_t::Coord_t;
  using typename Base_t::Weight_t;

  void Fill(const Coord_t &x, Weight_t weightN = 1.) {
    Base_t::Fill(x, weightN);
  }

  void FillN(const std::array_view<Coord_t> xN,
             const std::array_view<Weight_t> weightN) {
    Base_t::FillN(xN, weightN);
  }
  void FillN(const std::array_view<Coord_t> xN) {
    Base_t::FillN(xN);
  }
};


template<int DIMENSIONS, class PRECISION>
class THistStatRuntime: public THistStatEntries<DIMENSIONS, PRECISION> {
public:
  using Base_t = THistStatEntries<DIMENSIONS, PRECISION>;
  using typename Base_t::Coord_t;
  using typename Base_t::Weight_t;

  THistStatRuntime(bool uncertainty, std::vector<bool> &moments);

  void Fill(const Coord_t &x, Weight_t weightN = 1.) {
    Base_t::Fill(x, weightN);
  }

  void FillN(const std::array_view<Coord_t> xN,
             const std::array_view<Weight_t> weightN) {
    Base_t::FillN(xN, weightN);
  }
  void FillN(const std::array_view<Coord_t> xN) {
    Base_t::FillN(xN);
  }
};

} // namespace Experimental
} // namespace ROOT
#endif
