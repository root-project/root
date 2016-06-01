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
#include "ROOT/THistUtils.h"

namespace ROOT {
namespace Experimental {

template<int DIMENSIONS, class PRECISION,
  template <int D_, class P_, template <class P__> class STORAGE> class... STAT>
class THist;

/**
 \class THistStatContent
 Basic histogram statistics, keeping track of the bin content and the total
 number of calls to Fill().
 */
template<int DIMENSIONS, class PRECISION,
  template <class PRECISION_> class STORAGE>
class THistStatContent {
public:
  /// The type of a (possibly multi-dimensional) coordinate.
  using CoordArray_t = Hist::CoordArray_t<DIMENSIONS>;
  /// The type of the weight and the bin content.
  using Weight_t = PRECISION;
  /// Type of the bin content array.
  using Content_t = STORAGE<PRECISION>;

  /**
   \class TConstBinStat
   Const view on a THistStatContent for a given bin.
  */
  class TConstBinStat {
  public:
    TConstBinStat(const THistStatContent& stat, int index):
      fContent(stat.GetBinContent(index)) {}
    PRECISION GetContent() const { return fContent; }

  private:
    PRECISION fContent; ///< The content of this bin.
  };

  /**
   \class TBinStat
   Modifying view on a THistStatContent for a given bin.
  */
  class TBinStat {
  public:
    TBinStat(THistStatContent& stat, int index):
      fContent(stat.GetBinContent(index)) {}
    PRECISION& GetContent() const { return fContent; }

  private:
    PRECISION& fContent; ///< The content of this bin.
  };

  using ConstBinStat_t = TConstBinStat;
  using BinStat_t =  TBinStat;

private:
  /// Number of calls to Fill().
  int64_t fEntries = 0;

  /// Bin content.
  Content_t fBinContent;

public:
  THistStatContent() = default;
  THistStatContent(size_t in_size): fBinContent(in_size) {}

  /// Add weight to the bin content at binidx.
  void Fill(const CoordArray_t& /*x*/, int binidx, Weight_t weight = 1.) {
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
};

/**
 \class THistStatTotalSumOfWeights
 Keeps track of the histogram's total sum of weights.
 */
template<int DIMENSIONS, class PRECISION, template <class P_> class STORAGE>
class THistStatTotalSumOfWeights {
public:
  /// The type of a (possibly multi-dimensional) coordinate.
  using CoordArray_t = Hist::CoordArray_t<DIMENSIONS>;
  /// The type of the weight and the bin content.
  using Weight_t = PRECISION;

  /**
   \class TBinStat
   No-op; this class does not provide per-bin statistics.
  */
  template <bool>
  class TBinStat {
  public:
    TBinStat(const THistStatTotalSumOfWeights&, int) {}
  };

private:
  /// Sum of weights.
  PRECISION fSumWeights = 0;

public:
  THistStatTotalSumOfWeights() = default;
  THistStatTotalSumOfWeights(size_t) {}

  /// Add weight to the bin content at binidx.
  void Fill(const CoordArray_t& /*x*/, int, Weight_t weight = 1.) {
    fSumWeights += weight;
  }

  /// Get the sum of weights.
  Weight_t GetSumOfWeights() const { return fSumWeights; }

};


/**
 \class THistStatTotalSumOfSquaredWeights
 Keeps track of the histogram's total sum of squared weights.
 */
template<int DIMENSIONS, class PRECISION, template <class P_> class STORAGE>
class THistStatTotalSumOfSquaredWeights {
public:
  /// The type of a (possibly multi-dimensional) coordinate.
  using CoordArray_t = Hist::CoordArray_t<DIMENSIONS>;
  /// The type of the weight and the bin content.
  using Weight_t = PRECISION;

  /**
   \class TBinStat
   No-op; this class does not provide per-bin statistics.
  */
  class TBinStat {
  public:
    TBinStat(const THistStatTotalSumOfSquaredWeights&, int) {}
  };

  using ConstBinStat_t = TBinStat;
  using BinStat_t =  TBinStat;

private:
  /// Sum of (weights^2).
  PRECISION fSumWeights2 = 0;

public:
  THistStatTotalSumOfSquaredWeights() = default;
  THistStatTotalSumOfSquaredWeights(size_t) {}

  /// Add weight to the bin content at binidx.
  void Fill(const CoordArray_t& /*x*/, int /*binidx*/, Weight_t weight = 1.) {
    fSumWeights2 += weight * weight;
  }

  /// Get the sum of weights.
  Weight_t GetSumOfSquaredWeights() const { return fSumWeights2; }

};


/**
 \class THistStatUncertainty
 Histogram statistics to keep track of the Poisson uncertainty per bin.
 */
template<int DIMENSIONS, class PRECISION, template <class P_> class STORAGE>
class THistStatUncertainty {

public:
  /// The type of a (possibly multi-dimensional) coordinate.
  using CoordArray_t = Hist::CoordArray_t<DIMENSIONS>;
  /// The type of the weight and the bin content.
  using Weight_t = PRECISION;
  /// Type of the bin content array.
  using Content_t = STORAGE<PRECISION>;

  /**
   \class TConstBinStat
   Const view on a THistStatUncertainty for a given bin.
  */
  class TConstBinStat {
  public:
    TConstBinStat(const THistStatUncertainty& stat, int index):
      fSumW2(stat.GetSumOfSquaredWeights(index)) {}
    PRECISION GetSumW2() const { return fSumW2; }
    // Can never modify this. Set GetSumW2() instead.
    PRECISION GetUncertainty() const { return std::sqrt(std::abs(fSumW2)); }

  private:
    PRECISION fSumW2; ///< The bin's sum of square of weights.
  };

  /**
   \class TBinStat
   Modifying view on a THistStatUncertainty for a given bin.
  */
  class TBinStat {
  public:
    TBinStat(THistStatUncertainty& stat, int index):
      fSumW2(stat.GetSumOfSquaredWeights(index)) {}
    PRECISION& GetSumW2() const { return fSumW2; }
    // Can never modify this. Set GetSumW2() instead.
    PRECISION GetUncertainty() const { return std::sqrt(std::abs(fSumW2)); }

  private:
    PRECISION& fSumW2; ///< The bin's sum of square of weights.
  };

  using ConstBinStat_t = TConstBinStat;
  using BinStat_t =  TBinStat;

private:
  /// Uncertainty of the content for each bin.
  Content_t fSumWeightsSquared; ///< Sum of squared weights

public:
  THistStatUncertainty() = default;
  THistStatUncertainty(size_t size): fSumWeightsSquared(size) {}

  /// Add weight to the bin at binidx; the coordinate was x.
  void Fill(const CoordArray_t& /*x*/, int binidx, Weight_t weight = 1.) {
    fSumWeightsSquared[binidx] += weight * weight;
  }

  /// Calculate a bin's (Poisson) uncertainty of the bin content as the
  /// square-root of the bin's sum of squared weights.
  Weight_t GetBinUncertainty(int binidx) const {
    return std::sqrt(fSumWeightsSquared[binidx]);
  }

  /// Get a bin's sum of squared weights.
  Weight_t GetSumOfSquaredWeights(int binidx) const {
    return fSumWeightsSquared[binidx];
  }

  /// Get a bin's sum of squared weights.
  Weight_t& GetSumOfSquaredWeights(int binidx) {
    return fSumWeightsSquared[binidx];
  }

  /// Get the structure holding the sum of squares of weights.
  const std::vector<double>& GetSumOfSquaredWeights() const { return fSumWeightsSquared; }
  /// Get the structure holding the sum of squares of weights (non-const).
  std::vector<double>& GetSumOfSquaredWeights() { return fSumWeightsSquared; }
};

/** \class THistDataMomentUncert
  For now do as TH1: calculate first (xw) and second (x^2w) moment.
*/
template<int DIMENSIONS, class PRECISION, template <class P_> class STORAGE>
class THistDataMomentUncert {
public:
  /// The type of a (possibly multi-dimensional) coordinate.
  using CoordArray_t = Hist::CoordArray_t<DIMENSIONS>;
  /// The type of the weight and the bin content.
  using Weight_t = PRECISION;
  /// Type of the bin content array.
  using Content_t = STORAGE<PRECISION>;

  /**
   \class TBinStat
   No-op; this class does not provide per-bin statistics.
  */
  class TBinStat {
  public:
    TBinStat(const THistDataMomentUncert&, int) {}
  };

  using ConstBinStat_t = TBinStat;
  using BinStat_t =  TBinStat;

private:
  std::array<Weight_t, DIMENSIONS> fMomentXW;
  std::array<Weight_t, DIMENSIONS> fMomentX2W;

public:
  THistDataMomentUncert() = default;
  THistDataMomentUncert(size_t) {}

  /// Add weight to the bin at binidx; the coordinate was x.
  void Fill(const CoordArray_t &x, int /*binidx*/, Weight_t weight = 1.) {
    for (int idim = 0; idim < DIMENSIONS; ++idim) {
      const PRECISION xw = x[idim] * weight;
      fMomentXW[idim] += xw ;
      fMomentX2W[idim] += x[idim] * xw;
    }
  }
};


/** \class THistStatRuntime
  Interface implementing a pure virtual functions DoFill(), DoFillN().
  */
template<int DIMENSIONS, class PRECISION, template <class P_> class STORAGE>
class THistStatRuntime {
public:
  /// The type of a (possibly multi-dimensional) coordinate.
  using CoordArray_t = Hist::CoordArray_t<DIMENSIONS>;
  /// The type of the weight and the bin content.
  using Weight_t = PRECISION;
  /// Type of the bin content array.
  using Content_t = STORAGE<PRECISION>;

  /**
   \class TBinStat
   No-op; this class does not provide per-bin statistics.
  */
  class TBinStat {
  public:
    TBinStat(const THistStatRuntime&, int) {}
  };
  using ConstBinStat_t = TBinStat;
  using BinStat_t =  TBinStat;

  THistStatRuntime() = default;
  THistStatRuntime(size_t) {}
  virtual ~THistStatRuntime() = default;


  virtual void DoFill(const CoordArray_t &x, int binidx, Weight_t weightN) = 0;
  void Fill(const CoordArray_t &x, int binidx, Weight_t weight = 1.) {
    DoFill(x, binidx, weight);
  }
};


namespace Detail {

/// std::vector has more template arguments; for the default storage we don't
/// care about them, so use-decl them away:
template<class PRECISION>
using THistDataDefaultStorage = std::vector<PRECISION>;


template<int DIMENSIONS, class PRECISION,
  template <class P_> class STORAGE,
  template <int D_, class P_, template <class P__> class S_> class... STAT> class THistData;


/** \class TConstHistBinStat
  Const view on a bin's statistical data.
  */
template <int DIMENSIONS, class PRECISION,
  template <class P_> class STORAGE,
  template <int D_, class P_, template <class P__> class S_> class... STAT>
class TConstHistBinStat: public STAT<DIMENSIONS, PRECISION, STORAGE>::ConstBinStat_t... {
public:
  TConstHistBinStat(const THistData<DIMENSIONS, PRECISION, STORAGE, STAT...>& data, int index):
    STAT<DIMENSIONS, PRECISION, STORAGE>::ConstBinStat_t(data, index)... {}
};

/** \class THistBinStat
  Modifying view on a bin's statistical data.
  */
template <int DIMENSIONS, class PRECISION,
  template <class PRECISION_> class STORAGE,
  template <int D_, class P_, template <class P__> class S_> class... STAT>
class THistBinStat: public STAT<DIMENSIONS, PRECISION, STORAGE>::BinStat_t... {
public:
  THistBinStat(THistData<DIMENSIONS, PRECISION, STORAGE, STAT...>& data, int index):
    STAT<DIMENSIONS, PRECISION, STORAGE>::BinStat_t(data, index)... {}
};


/** \class THistData
  A THistImplBase's data, provides accessors to all its statistics.
  */
template<int DIMENSIONS, class PRECISION,
  template <class P_> class STORAGE,
  template <int D_, class P_, template <class P__> class S_> class... STAT>
class THistData: public STAT<DIMENSIONS, PRECISION, STORAGE>... {
private:
  template <class T>
  static auto HaveGetBinUncertainty(THistData* This) -> decltype(This->GetBinUncertainty(12))
  { return 0; }
  template <class T> static int HaveGetBinUncertainty(...) { return 0; }
  static constexpr const bool fgHaveGetBinUncertainty
    = sizeof(HaveGetBinUncertainty<THistData>(nullptr)) == sizeof(char);

public:
  /// Matching THist
  using Hist_t = THist<DIMENSIONS, PRECISION, STAT...>;

  /// The type of the weight and the bin content.
  using Weight_t = PRECISION;

  /// The type of a (possibly multi-dimensional) coordinate.
  using CoordArray_t = Hist::CoordArray_t<DIMENSIONS>;

  /// The type of a non-modifying view on a bin.
  using ConstHistBinStat_t
    = TConstHistBinStat<DIMENSIONS, PRECISION, STORAGE, STAT...>;

  /// The type of a modifying view on a bin.
  using HistBinStat_t
    = THistBinStat<DIMENSIONS, PRECISION, STORAGE, STAT...>;

  THistData() = default;


  /// Constructor providing the number of bins (incl under, overflow) to the
  /// base classes.
  THistData(size_t size): STAT<DIMENSIONS, PRECISION, STORAGE>(size)... {}

  /// Fill weight at x to the bin content at binidx.
  void Fill(const CoordArray_t& x, int binidx, Weight_t weight = 1.) {
    // Call Fill() on all base classes.
    using expand_type = int[];
    (void)expand_type{ (STAT<DIMENSIONS, PRECISION, STORAGE>::Fill(x, binidx, weight), 0)... };
  }

  /// Get a view on the statistics values of a bin.
  ConstHistBinStat_t GetView(int idx) const {
    return ConstHistBinStat_t(*this, idx);
  }
  /// Get a (non-const) view on the statistics values of a bin.
  HistBinStat_t GetView(int idx) {
    return HistBinStat_t(*this, idx);
  }

  /// Calculate the bin content's uncertainty for the given bin, using Poisson
  /// statistics on the absolute bin content. Only available if no base provides
  /// this functionality. Requires GetBinContent(int binIndex).
  template <class T = typename std::enable_if<!fgHaveGetBinUncertainty>::type>
  Weight_t GetBinUncertainty(int binidx) const {
    return std::sqrt(std::fabs(this->GetBinContent(binidx)));
  }

};
} // namespace Detail

} // namespace Experimental
} // namespace ROOT
#endif
