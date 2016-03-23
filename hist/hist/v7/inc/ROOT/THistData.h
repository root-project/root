/// \file ROOT/THistData.h
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

#ifndef ROOT7_THistData_h
#define ROOT7_THistData_h

#include <cmath>
#include <vector>
#include "ROOT/RArrayView.h"

namespace ROOT {
namespace Experimental {

/// std::vector has more template arguments; for the default storage we don't
/// care about them, so use-decl them away:
template<class PRECISION> using THistDataDefaultStorage
  = std::vector<PRECISION>;

/**
 \class THistDataContent
 Basic histogram statistics, keeping track of the bin content and the total
 number of calls to Fill().
 */
template<int DIMENSIONS, class PRECISION,
  template <class PRECISION_> class STORAGE = THistDataDefaultStorage>
class THistDataContent {
public:
  /// The type of a (possibly multi-dimensional) coordinate.
  using Coord_t = std::array<double, DIMENSIONS>;
  /// The type of the weight and the bin content.
  using Weight_t = PRECISION;
  /// Type of the bin content array.
  using Content_t = STORAGE<PRECISION>;

  /**
   \class TBinStat
   Mutable view on a THistDataContent for a given bin.
   Template argument can be `Weight_t` for a const view or `Weight_t&` for a
   modifying view.
  */
  template <class WEIGHT>
  class TBinStat {
  public:
    TBinStat(WEIGHT content): fContent(content) {}
    WEIGHT GetContent() const { return fContent; }

  private:
    WEIGHT fContent; ///< (Reference to) the content of this bin.
  };

private:
  /// Number of calls to Fill().
  int64_t fEntries = 0;

  /// Bin content.
  Content_t fBinContent;

public:
  THistDataContent(size_t size): fBinContent(size) {}

  /// Add weight to the bin content at binidx.
  void Fill(const Coord_t& /*x*/, int binidx, Weight_t weight = 1.) {
    fBinContent[binidx] += weight;
    ++fEntries;
  }

  /// Number of dimensions of the coordinates
  static constexpr int GetNDim() noexcept { return DIMENSIONS; }

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
  TBinStat<Weight_t> GetView(int idx) const {
    return TBinStat<Weight_t>(fBinContent[idx]);
  }
  /// Get a (non-const) view on the statistics values of a bin.
  TBinStat<Weight_t&> GetView(int idx) {
    return TBinStat<Weight_t&>(fBinContent[idx]);
  }
};


/**
 \class THistDataUncertainty
 Histogram statistics to keep track of the bin content and its Poisson
 uncertainty per bin, and the total number of calls to Fill().
 */
template<int DIMENSIONS, class PRECISION,
  template <class PRECISION_> class STORAGE = THistDataDefaultStorage>
class THistDataUncertainty: public THistDataContent<DIMENSIONS, PRECISION, STORAGE> {

public:
  using Base_t = THistDataContent<DIMENSIONS, PRECISION, STORAGE>;
  using typename Base_t::Coord_t;
  using typename Base_t::Weight_t;
  using typename Base_t::Content_t;

  /**
   \class TBinStat
   View on a THistDataUncertainty for a given bin.
   Template argument can be `Weight_t` for a const view or `Weight_t&` for a
   modifying view.
  */
  template <class WEIGHT>
  class TBinStat {
  public:
    TBinStat(WEIGHT content, WEIGHT sumw2):
      fContent(content), fSumW2(sumw2) {}
    WEIGHT GetContent() const { return fContent; }
    WEIGHT GetSumW2() const { return fSumW2; }
    // Can never modify this. Set GetSumW2() instead.
    WEIGHT GetUncertainty() const { return std::sqrt(std::abs(fSumW2)); }

  private:
    WEIGHT fContent; ///< Content of this bin.
    WEIGHT fSumW2; ///< The bin's sum of square of weights.
  };

private:
  /// Uncertainty of the content for each bin.
  Content_t fSumWeightsSquared; ///< Sum of squared weights

public:
  THistDataUncertainty(size_t size): Base_t(size), fSumWeightsSquared(size) {}

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
  TBinStat<Weight_t> GetView(int idx) const {
    return TBinStat<Weight_t>(this->GetBinContent(idx), fSumWeightsSquared[idx]);
  }
  /// Get a (non-const) view on the statistics values of a bin.
  TBinStat<Weight_t&> GetView(int idx) {
    return TBinStat<Weight_t&>(this->GetBinContent(idx), fSumWeightsSquared[idx]);
  }
};

template<int DIMENSIONS, class PRECISION,
  template <class PRECISION_> class STORAGE = THistDataDefaultStorage>
class THistDataMomentUncert:
  public THistDataUncertainty<DIMENSIONS, PRECISION, STORAGE> {
public:
  using Base_t = THistDataUncertainty<DIMENSIONS, PRECISION, STORAGE>;
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
  template <class PRECISION_> class STORAGE = THistDataDefaultStorage>
class THistDataRuntime:
  public THistDataContent<DIMENSIONS, PRECISION, STORAGE> {
public:
  using Base_t = THistDataContent<DIMENSIONS, PRECISION, STORAGE>;
  using typename Base_t::Coord_t;
  using typename Base_t::Weight_t;

  THistDataRuntime(bool uncertainty, std::vector<bool> &moments);

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
