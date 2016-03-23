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

/// std::vector has more template arguments; for the default storage we don't
/// care about them, so use-decl them away:
template<class PRECISION> using THistStatDefaultStorage
  = std::vector<PRECISION>;

/**
 \class THistStatContent
 Basic histogram statistics, keeping track of the bin content and the total
 number of calls to Fill().
 */
template<int DIMENSIONS, class PRECISION,
  template <class PRECISION_> class STORAGE = THistStatDefaultStorage>
class THistStatContent {
public:
  /// The type of a (possibly multi-dimensional) coordinate.
  using Coord_t = std::array<double, DIMENSIONS>;
  /// The type of the weight and the bin content.
  using Weight_t = PRECISION;
  /// Type of the bin content array.
  using Content_t = STORAGE<PRECISION>;

  /**
  Const-view on a THistStatContent for a given bin.
  */
  class TConstBinStat {
  public:
    TConstBinStat(Weight_t content): fContent(content) {}
    Weight_t GetContent() const { return fContent; }

  private:
    Weight_t fContent; ///< Content of this bin.
  };

  /**
  Mutable view on a THistStatContent for a given bin.
  */
  class TBinStat {
  public:
    TBinStat(Weight_t& content): fContent(content) {}
    Weight_t& GetContent() const { return fContent; }

  private:
    Weight_t& fContent; ///< Reference to the content of this bin.
  };

private:
  /// Number of calls to Fill().
  int64_t fEntries = 0;

  /// Bin content.
  Content_t fBinContent;

public:
  THistStatContent(size_t size): fBinContent(size) {}

  /// Add weight to the bin content at binidx.
  void Fill(const Coord_t& /*x*/, int binidx, Weight_t weight = 1.) {
    fBinContent[binidx] += weight;
    ++fEntries;
  }

  /// Get the number of entries filled into the histogram - i.e. the number of
  /// calls to Fill().
  int64_t GetEntries() const { return fEntries; }

  /// Get the number of bins.
  size_t size() const noexcept { return fBinContent.size(); }

  /// Get the bin content for the given bin.
  Weight_t operator[](int idx) const { return fBinContent[idx]; }
  /// Get the bin content for the given bin (non-const).
  Weight_t& operator[](int idx) { return fBinContent[idx]; }

  /// Get the bin content for the given bin.
  Weight_t GetBinContent(int idx) const { return fBinContent[idx]; }
  /// Get the bin content for the given bin (non-const).
  Weight_t& GetBinContent(int idx) { return fBinContent[idx]; }

  /// Retrieve the content array.
  const Content_t& GetContentArray() const { return fBinContent; }
  /// Retrieve the content array (non-const).
  Content_t& GetContentArray() { return fBinContent; }

  /// Calculate the bin content's uncertainty for the given bin, using Poisson
  /// statistics on the absolute bin content.
  Weight_t GetBinUncertainty(int binidx) const {
    return std::sqrt(std::fabs(fBinContent[binidx]));
  }

  /// Get a view on the statistics values of a bin.
  TConstBinStat GetView(int idx) const { return TConstBinStat(fBinContent[idx]); }
  /// Get a (non-const) view on the statistics values of a bin.
  TBinStat GetView(int idx) { return TBinStat(fBinContent[idx]); }
};


/**
 \class THistStatUncertainty
 Histogram statistics to keep track of the bin content and its Poisson
 uncertainty per bin, and the total number of calls to Fill().
 */
template<int DIMENSIONS, class PRECISION,
  template <class PRECISION_> class STORAGE = THistStatDefaultStorage>
class THistStatUncertainty: public THistStatContent<DIMENSIONS, PRECISION, STORAGE> {

public:
  using Base_t = THistStatContent<DIMENSIONS, PRECISION, STORAGE>;
  using typename Base_t::Coord_t;
  using typename Base_t::Weight_t;
  using typename Base_t::Content_t;

  /**
  Const-view on a THistStatUncertainty for a given bin.
  */
  class TConstBinStat {
  public:
    TConstBinStat(Weight_t content, Weight_t sumw2):
      fContent(content), fSumW2(sumw2) {}
    Weight_t GetContent() const { return fContent; }
    Weight_t GetSumW2() const { return fSumW2; }

  private:
    Weight_t fContent; ///< Content of this bin.
    Weight_t fSumW2; ///< The bin's sum of square of weights.
  };

  /**
  Mutable view on a THistStatUncertainty for a given bin.
  */
  class TBinStat {
  public:
    TBinStat(Weight_t& content, Weight_t& sumw2):
    fContent(content), fSumW2(sumw2) {}
    Weight_t& GetContent() const { return fContent; }
    Weight_t& GetSumW2() const { return fSumW2; }

  private:
    Weight_t& fContent; ///< Content of this bin.
    Weight_t& fSumW2; ///< Uncertainty of the content of this bin.
  };


private:
  /// Uncertainty of the content for each bin.
  Content_t fSumWeightsSquared; ///< Sum of squared weights

public:
  THistStatUncertainty(size_t size): Base_t(size), fSumWeightsSquared(size) {}

  /// Add weight to the bin at binidx; the coordinate was x.
  void Fill(const Coord_t &x, int binidx, Weight_t weight = 1.) {
    Base_t::Fill(x, binidx, weight);
    fSumWeightsSquared[binidx] += weight * weight;
  }

  /// Calculate a bin's (Poisson) uncertainty of the bin content as the
  /// square-root of the bin's sum of squared weights.
  Weight_t GetBinUncertainty(int binidx) const {
    return std::sqrt(fSumWeightsSquared[binidx]);
  }

  /// Get the structure holding the sum of squares of weights.
  const std::vector<double>& GetSumWeightsSquared() const { return fSumWeightsSquared; }
  /// Get the structure holding the sum of squares of weights (non-const).
  std::vector<double>& GetSumWeightsSquared() { return fSumWeightsSquared; }

  /// Get a view on the statistics values of a bin.
  TConstBinStat GetView(int idx) const {
    return TConstBinStat(this->GetBinContent(idx), fSumWeightsSquared[idx]);
  }
  /// Get a (non-const) view on the statistics values of a bin.
  TBinStat GetView(int idx) {
    return TBinStat(this->GetBinContent(idx), fSumWeightsSquared[idx]);
  }
};

template<int DIMENSIONS, class PRECISION,
  template <class PRECISION_> class STORAGE = THistStatDefaultStorage>
class THistStatMomentUncert:
  public THistStatUncertainty<DIMENSIONS, PRECISION, STORAGE> {
public:
  using Base_t = THistStatUncertainty<DIMENSIONS, PRECISION, STORAGE>;
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

template<int DIMENSIONS, class PRECISION,
  template <class PRECISION_> class STORAGE = THistStatDefaultStorage>
class THistStatRuntime:
  public THistStatContent<DIMENSIONS, PRECISION, STORAGE> {
public:
  using Base_t = THistStatContent<DIMENSIONS, PRECISION, STORAGE>;
  using typename Base_t::Coord_t;
  using typename Base_t::Weight_t;

  THistStatRuntime(bool uncertainty, std::vector<bool> &moments);

  virtual void DoFill(const Coord_t &x, Weight_t weightN) = 0;
  void Fill(const Coord_t &x, Weight_t weight = 1.) {
    Base_t::Fill(x, weight);
    DoFill(x, weight);
  }

  virtual void DoFillN(const std::array_view<Coord_t> xN,
                       const std::array_view<Weight_t> weightN) = 0;
  void FillN(const std::array_view<Coord_t> xN,
             const std::array_view<Weight_t> weightN) {
    Base_t::FillN(xN, weightN);
    DoFill(xN, weightN);
  }

  virtual void DoFillN(const std::array_view<Coord_t> xN) = 0;
  void FillN(const std::array_view<Coord_t> xN) {
    Base_t::FillN(xN);
    DoFill(xN);
  }
};

} // namespace Experimental
} // namespace ROOT
#endif
