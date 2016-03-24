/// \file ROOT/THist.h
/// \ingroup Hist ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-03-23
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_THist
#define ROOT7_THist

#include "ROOT/RArrayView.h"
#include "ROOT/TAxis.h"
#include "ROOT/TDrawable.h"
#include "ROOT/THistBinIter.h"
#include "ROOT/THistDrawable.h"
#include "ROOT/THistImpl.h"
#include "ROOT/THistData.h"
#include <initializer_list>

namespace ROOT {
namespace Experimental {

// fwd declare for fwd declare for friend declaration in THist...
template<class DATA> class THist;

// fwd declare for friend declaration in THist.
template<class DATA>
class THist<DATA>
  HistFromImpl(std::unique_ptr<typename THist<DATA>::ImplBase_t> pHistImpl);

template<class DATA>
void swap(THist<DATA> &a, THist<DATA> &b) noexcept;

/**
 \class THist
 Histogram class for histograms with `DATA::GetNDim()` dimensions, where each
 bin count is stored by a value of type `DATA::Weight_t`. DATA stores statistical
 data of the entries filled into the histogram (bin content, uncertainties etc).

 A histogram counts occurrences of values or n-dimensional combinations thereof.
 Contrary to for instance a `TTree`, a histogram combines adjacent values. The
 resolution of this combination is defined by the axis binning, see e.g.
 http://www.wikiwand.com/en/Histogram
 */

template<class DATA>
class THist {
public:
  /// The type of the `Detail::THistImplBase` of this histogram.
  using ImplBase_t = Detail::THistImplBase<DATA>;
  /// The coordinates type: a `GetNDim()`-dimensional `std::array` of `double`.
  using Coord_t = typename ImplBase_t::Coord_t;
  /// Statistics type
  using Stat_t = DATA;
  /// The type of weights
  using Weight_t = typename Stat_t::Weight_t;
  /// Pointer type to `HistImpl_t::Fill`, for faster access.
  using FillFunc_t = typename ImplBase_t::FillFunc_t;

  using const_iterator = Detail::THistBinIter<ImplBase_t>;

  /// Number of dimensions of the coordinates
  static constexpr int GetNDim() noexcept { return DATA::GetNDim(); }

  THist() = default;
  THist(THist&&) = default;

  /// Create a histogram from an `array` of axes (`TAxisConfig`s). Example code:
  ///
  /// Construct a 1-dimensional histogram that can be filled with `floats`s.
  /// The axis has 10 bins between 0. and 1. The two outermost sets of curly
  /// braces are to reach the initialization of the `std::array` elements; the
  /// inner one is for the initialization of a `TAxisCoordinate`.
  ///
  ///     THist<1,float> h1f({{ {10, 0., 1.} }});
  ///
  /// Construct a 2-dimensional histogram, with the first axis as before, and
  /// the second axis having non-uniform ("irregular") binning, where all bin-
  /// edges are specified. As this is itself an array it must be enclosed by
  /// double curlies.
  ///
  ///     THist<2,int> h2i({{ {10, 0., 1.}, {{-1., 0., 1., 10., 100.}} }});
  explicit THist(std::array<TAxisConfig, THist::GetNDim()> axes);

  /// Constructor overload taking the histogram title
  THist(std::string_view histTitle, std::array<TAxisConfig, THist::GetNDim()> axes);

  /// Constructor overload that's only available for a 1-dimensional histogram.
  template<int ENABLEIF_NDIM = DATA::GetNDim(),
    class = typename std::enable_if<ENABLEIF_NDIM == 1>::type>
  THist(const TAxisConfig &xaxis):
    THist(std::array<TAxisConfig, 1>{{xaxis}}) { }

  /// Constructor overload that's only available for a 1-dimensional histogram,
  /// also passing the histogram title.
  template<int ENABLEIF_NDIM = DATA::GetNDim(),
    class = typename std::enable_if<ENABLEIF_NDIM == 1>::type>
  THist(std::string_view histTitle, const TAxisConfig &xaxis):
    THist(histTitle, std::array<TAxisConfig, 1>{{xaxis}}) { }

  /// Constructor overload that's only available for a 2-dimensional histogram.
  template<int ENABLEIF_NDIM = DATA::GetNDim(),
    class = typename std::enable_if<ENABLEIF_NDIM == 2>::type>
  THist(const TAxisConfig &xaxis, const TAxisConfig &yaxis):
    THist(std::array<TAxisConfig, 2>{{xaxis, yaxis}}) { }

  /// Constructor overload that's only available for a 2-dimensional histogram,
  /// also passing the histogram title.
  template<int ENABLEIF_NDIM = DATA::GetNDim(),
    class = typename std::enable_if<ENABLEIF_NDIM == 2>::type>
  THist(std::string_view histTitle, const TAxisConfig &xaxis, const TAxisConfig &yaxis):
    THist(histTitle, std::array<TAxisConfig, 2>{{xaxis, yaxis}}) { }

  /// Constructor overload that's only available for a 3-dimensional histogram.
  template<int ENABLEIF_NDIM = DATA::GetNDim(),
    class = typename std::enable_if<ENABLEIF_NDIM == 3>::type>
  THist(const TAxisConfig &xaxis, const TAxisConfig &yaxis,
        const TAxisConfig &zaxis):
    THist(std::array<TAxisConfig, 3>{{xaxis, yaxis, zaxis}}) { }

  /// Constructor overload that's only available for a 3-dimensional histogram,
  /// also passing the histogram title.
  template<int ENABLEIF_NDIM = DATA::GetNDim(),
    class = typename std::enable_if<ENABLEIF_NDIM == 3>::type>
  THist(std::string_view histTitle,
        const TAxisConfig &xaxis, const TAxisConfig &yaxis,
        const TAxisConfig &zaxis):
    THist(histTitle, std::array<TAxisConfig, 3>{{xaxis, yaxis, zaxis}}) { }


  /// Access the ImplBase_t this THist points to.
  ImplBase_t *GetImpl() const noexcept { return fImpl.get(); }

  /// Add `weight` to the bin containing coordinate `x`.
  void Fill(const Coord_t &x, Weight_t weight = (Weight_t) 1) noexcept {
    (fImpl.get()->*fFillFunc)(x, weight);
  }

  /// For each coordinate in `xN`, add `weightN[i]` to the bin at coordinate
  /// `xN[i]`. The sizes of `xN` and `weightN` must be the same. This is more
  /// efficient than many separate calls to `Fill()`.
  void FillN(const std::array_view <Coord_t> xN,
             const std::array_view <Weight_t> weightN) noexcept {
    fImpl->FillN(xN, weightN);
  }

  /// Convenience overload: `FillN()` with weight 1.
  void FillN(const std::array_view <Coord_t> xN) noexcept {
    fImpl->FillN(xN);
  }

  /// Get the number of entries this histogram was filled with.
  int64_t GetEntries() const noexcept { return fImpl->GetStat().GetEntries(); }

  /// Get the content of the bin at `x`.
  Weight_t GetBinContent(const Coord_t &x) const {
    return fImpl->GetBinContent(x);
  }

  /// Get the uncertainty on the content of the bin at `x`.
  Weight_t GetBinUncertainty(const Coord_t &x) const {
    return fImpl->GetBinUncertainty(x);
  }

  const_iterator begin() const { return const_iterator(*fImpl); }

  const_iterator end() const { return const_iterator(*fImpl, fImpl->GetNBins()); }

private:
  std::unique_ptr<ImplBase_t> fImpl; ///< The actual histogram implementation
  FillFunc_t fFillFunc = nullptr; ///< Pinter to THistImpl::Fill() member function

  friend THist HistFromImpl<>(std::unique_ptr<ImplBase_t>);
  friend void swap<>(THist<DATA> &a, THist<DATA> &b) noexcept;

};

/// Swap two histograms.
///
/// Very efficient; swaps the `fImpl` pointers.
template<class DATA>
void swap(THist<DATA> &a, THist<DATA> &b) noexcept {
  std::swap(a.fImpl, b.fImpl);
  std::swap(a.fFillFunc, b.fFillFunc);
};


/// Adopt an external, stand-alone THistImpl. The THist will take ownership.
template<class DATA>
THist<DATA> HistFromImpl(std::unique_ptr<typename THist<DATA>::ImplBase_t> pHistImpl) {
  THist<DATA> ret;
  ret.fFillFunc = pHistImpl->GetFillFunc();
  std::swap(ret.fImpl, pHistImpl);
  return ret;
};


namespace Internal {
/**
 Generate THist::fImpl from THist constructor arguments.
 */
template<int NDIM, int IDIM, class DATA, class... PROCESSEDAXISCONFIG>
struct HistImplGen_t {
  /// Select the template argument for the next axis type, and "recurse" into
  /// HistImplGen_t for the next axis.
  template<TAxisConfig::EKind KIND>
  std::unique_ptr<Detail::THistImplBase<DATA>>
  MakeNextAxis(std::string_view title,
               const std::array<TAxisConfig, NDIM> &axes,
               PROCESSEDAXISCONFIG... processedAxisArgs) {
    typename AxisConfigToType<KIND>::Axis_t nextAxis
      = AxisConfigToType<KIND>()(axes[IDIM]);
    return HistImplGen_t<NDIM, IDIM + 1, DATA,
      PROCESSEDAXISCONFIG..., typename AxisConfigToType<KIND>::Axis_t>()
      (title, axes, processedAxisArgs..., nextAxis);
  }

  /// Make a THistImpl-derived object reflecting the TAxisConfig array.
  ///
  /// Delegate to the appropriate MakeNextAxis instantiation, depending on the
  /// axis type selected in the TAxisConfig.
  /// \param axes - `TAxisConfig` objects describing the axis of the resulting
  ///   THistImpl.
  /// \param statConfig - the statConfig parameter to be passed to the THistImpl
  /// \param processedAxisArgs - the TAxisBase-derived axis objects describing the
  ///   axes of the resulting THistImpl. There are `IDIM` of those; in the end
  /// (`IDIM` == `GetNDim()`), all `axes` have been converted to
  /// `processedAxisArgs` and the THistImpl constructor can be invoked, passing
  /// the `processedAxisArgs`.
  std::unique_ptr<Detail::THistImplBase<DATA>>
  operator()(std::string_view title,
             const std::array <TAxisConfig, NDIM> &axes,
             PROCESSEDAXISCONFIG... processedAxisArgs) {
    switch (axes[IDIM].GetKind()) {
      case TAxisConfig::kEquidistant:
        return MakeNextAxis<TAxisConfig::kEquidistant>(title, axes,
                                                       processedAxisArgs...);
      case TAxisConfig::kGrow:
        return MakeNextAxis<TAxisConfig::kGrow>(title, axes,
                                                processedAxisArgs...);
      case TAxisConfig::kIrregular:
        return MakeNextAxis<TAxisConfig::kIrregular>(title, axes,
                                                     processedAxisArgs...);
      default:
        R__ERROR_HERE("HIST") << "Unhandled axis kind";
    }
    return nullptr;
  }
};

/// Generate THist::fImpl from constructor arguments; recursion end.
template<int NDIM, class DATA, class... PROCESSEDAXISCONFIG>
/// Create the histogram, now that all axis types and initializer objects are
/// determined.
struct HistImplGen_t<NDIM, NDIM, DATA, PROCESSEDAXISCONFIG...> {
  using HistImplBase_t = ROOT::Experimental::Detail::THistImplBase<DATA>;
  std::unique_ptr<HistImplBase_t>
  operator()(std::string_view title, const std::array<TAxisConfig, DATA::GetNDim()> &,
             PROCESSEDAXISCONFIG... axisArgs) {
    using HistImplt_t
    = Detail::THistImpl<DATA, PROCESSEDAXISCONFIG...>;
    return std::make_unique<HistImplt_t>(title, axisArgs...);
  }
};
} // namespace Internal


template<class DATA>
THist<DATA>::THist(std::array<TAxisConfig, THist::GetNDim()> axes):
  fImpl{std::move(Internal::HistImplGen_t<THist::GetNDim(), 0, DATA>()("", axes))},
  fFillFunc{} {
  fFillFunc = fImpl->GetFillFunc();
}

template<class DATA>
THist<DATA>::THist(std::string_view title,
                   std::array<TAxisConfig, THist::GetNDim()> axes):
  fImpl{std::move(Internal::HistImplGen_t<THist::GetNDim(), 0, DATA>()(title, axes))},
  fFillFunc{} {
  fFillFunc = fImpl->GetFillFunc();
}

/// \name THist Typedefs
///\{ Convenience typedefs (ROOT6-compatible type names)

// Keep them as typedefs, to make sure old-style documentation tools can
// understand them.
typedef THist<THistDataUncertainty<1, double>> TH1D;
typedef THist<THistDataUncertainty<1, float>> TH1F;
typedef THist<THistDataContent<1, char>> TH1C;
typedef THist<THistDataContent<1, int>> TH1I;
typedef THist<THistDataContent<1, int64_t>> TH1LL;

typedef THist<THistDataUncertainty<2, double>> TH2D;
typedef THist<THistDataUncertainty<2, float>> TH2F;
typedef THist<THistDataContent<2, char>> TH2C;
typedef THist<THistDataContent<2, int>> TH2I;
typedef THist<THistDataContent<2, int64_t>> TH2LL;

typedef THist<THistDataUncertainty<3, double>> TH3D;
typedef THist<THistDataUncertainty<3, float>> TH3F;
typedef THist<THistDataContent<3, char>> TH3C;
typedef THist<THistDataContent<3, int>> TH3I;
typedef THist<THistDataContent<3, int64_t>> TH3LL;
///\}


/// Add two histograms. This is the generic, inefficient version for now; it
/// assumes no matching axes.
template<class DATAA, class DATAB,
  class = typename std::enable_if<"Cannot add histograms with different number of dimensions!"
                                  && DATAA::GetNDim() == DATAB::GetNDim()>::type>
void Add(THist<DATAA> &to,THist<DATAB> &from) {
  auto toImpl = to.GetImpl();
  auto fillFuncTo = toImpl->GetFillFunc();
  using FromCoord_t = typename THist<DATAB>::Coord_t;
  using FromWeight_t = typename THist<DATAB>::Weight_t;
  auto add = [fillFuncTo, toImpl](const FromCoord_t& x, FromWeight_t c) {
    (toImpl->*fillFuncTo)(x, c);
    // TODO: something nice with the uncertainty - depending on whether `to` cares
  };
  from.GetImpl()->ApplyXC(add);
};


/// Interface to graphics taking a unique_ptr<THist>.
template<class DATA>
std::unique_ptr <Internal::TDrawable>
GetDrawable(std::shared_ptr<THist<DATA>> hist,
            THistDrawOptions<DATA::GetNDim()> opts = {}) {
  return std::make_unique<Internal::THistDrawable<DATA>>(hist, opts);
}

/// Interface to graphics taking a shared_ptr<THist>.
template<class DATA>
std::unique_ptr <Internal::TDrawable>
GetDrawable(std::unique_ptr<THist<DATA>> hist,
            THistDrawOptions<DATA::GetNDim()> opts = {}) {
  return std::make_unique<Internal::THistDrawable<DATA>>(hist, opts);
}

} // namespace Experimental
} // namespace ROOT

#endif
