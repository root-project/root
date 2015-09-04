/// \file ROOT/THist.h
/// \ingroup Hist
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-03-23

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
#include "ROOT/THistStats.h"
#include "ROOT/TCoopPtr.h"
#include <initializer_list>

namespace ROOT {

// fwd declare for friend declaration in THist.
template <int DIMENSIONS, class PRECISION>
THist<DIMENSIONS, PRECISION>
HistFromImpl(std::unique_ptr<typename THist<DIMENSIONS, PRECISION>::ImplBase_t> pHistImpl);

template <int DIMENSIONS, class PRECISION>
void swap(THist<DIMENSIONS, PRECISION> &a,
          THist<DIMENSIONS, PRECISION> &b) noexcept;


/**
 \class THist
 Histogram class for histograms with `DIMENSIONS` dimensions, where each bin
 count is stored by a value of type `PRECISION`.

 A histogram counts occurrences of values or n-dimensional combinations thereof.
 Contrary to for instance a `TTree`, a histogram combines adjacent values. The
 resolution of this combination is defined by the binning, see e.g.
 http://www.wikiwand.com/en/Histogram
 */

template<int DIMENSIONS, class PRECISION>
class THist {
public:
  /// The type of the `THistImplBase` of this histogram.
  using ImplBase_t = THistImplBase<DIMENSIONS, PRECISION>;
  /// The coordinates type: a `DIMENSIONS`-dimensional `std::array` of `double`.
  using Coord_t = typename ImplBase_t::Coord_t;
  /// The type of weights (`PRECISION`)
  using Weight_t = typename ImplBase_t::Weight_t;
  /// Pointer type to `HistImpl_t::Fill`, for faster access.
  using FillFunc_t = typename ImplBase_t::FillFunc_t;

  using const_iterator = Internal::THistBinIter<Internal::HistIterFullRange_t>;

  THist() = default;

  /// Create a histogram from an `array` of axes (`TAxisConfig`s) and possibly
  /// an initial `STATISTICS` object. The latter is usually just fine when
  /// not passed (i.e. default-constructed). Example code:
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
  template<class STATISTICS = THistStatUncertainty<DIMENSIONS, PRECISION>>
  THist(std::array<TAxisConfig, DIMENSIONS> axes,
        STATISTICS statConfig = STATISTICS());

  template<class STATISTICS = THistStatUncertainty<DIMENSIONS, PRECISION>,
           class = typename std::enable_if<DIMENSIONS == 1 &&
                                           std::is_default_constructible<STATISTICS>::value>::type>
  THist(const TAxisConfig& xaxis, STATISTICS statConfig = STATISTICS()):
    THist({{xaxis}}, statConfig) {}

  template<class STATISTICS = THistStatUncertainty<DIMENSIONS, PRECISION>,
     class = typename std::enable_if<DIMENSIONS == 2 &&
                                     std::is_default_constructible<STATISTICS>::value>::type>
  THist(const TAxisConfig& xaxis, const TAxisConfig& yaxis,
     STATISTICS statConfig = STATISTICS()):
    THist({{xaxis, yaxis}}, statConfig) {}

  template<class STATISTICS = THistStatUncertainty<DIMENSIONS, PRECISION>,
     class = typename std::enable_if<DIMENSIONS == 3 &&
                                     std::is_default_constructible<STATISTICS>::value>::type>
  THist(const TAxisConfig& xaxis, const TAxisConfig& yaxis,
        const TAxisConfig& zaxis, STATISTICS statConfig = STATISTICS()):
  THist({{xaxis, yaxis, zaxis}}, statConfig) {}


  /// Number of dimensions of the coordinates
  static constexpr int GetNDim() noexcept { return DIMENSIONS; }

  /// Access the ImplBase_t this THist points to.
  ImplBase_t* GetImpl() const noexcept { return fImpl.get(); }

  /// Add `weight` to the bin containing coordinate `x`.
  void Fill(const Coord_t &x, Weight_t weight = 1.) noexcept {
    (fImpl.get()->*fFillFunc)(x, weight);
  }

  /// For each coordinate in `xN`, add `weightN[i]` to the bin at coordinate
  /// `xN[i]`. The sizes of `xN` and `weightN` must be the same. This is more
  /// efficient than many separate calls to `Fill()`.
  void FillN(const std::array_view<Coord_t> xN,
             const std::array_view<Weight_t> weightN) noexcept {
    fImpl->FillN(xN, weightN);
  }

  /// Convenience overload: `FillN()` with weight 1.
  void FillN(const std::array_view<Coord_t> xN) noexcept {
    fImpl->FillN(xN);
  }

  /// Get the number of entries this histogram was filled with.
  int64_t GetEntries() const noexcept { return fImpl->GetStat().GetEntries(); }

  const_iterator begin() const { return const_iterator(0); }
  const_iterator end() const { return const_iterator(fImpl->GetNBins()); }

private:
  FillFunc_t fFillFunc = nullptr; ///< Pinter to THistImpl::Fill() member function
  std::unique_ptr<ImplBase_t> fImpl; ///< The actual histogram implementation

  friend THist HistFromImpl<>(std::unique_ptr<ImplBase_t>);
  friend void swap<>(THist<DIMENSIONS, PRECISION> &a,
                     THist<DIMENSIONS, PRECISION> &b) noexcept;

};

/// Swap two histograms.
///
/// Very efficient; swaps the `fImpl` pointers.
template <int DIMENSIONS, class PRECISION>
void swap(THist<DIMENSIONS, PRECISION> &a,
          THist<DIMENSIONS, PRECISION> &b) noexcept {
  std::swap(a.fImpl, b.fImpl);
  std::swap(a.fFillFunc, b.fFillFunc);
};


/// Create a TCoopPtr of a THist.
//
// As make_xyz cannot deal with initializer_lists, we need to expose THist's
// constructor arguments and take THist's template arguments.
template <int DIMENSIONS, class PRECISION, class STATISTICS = THistStatUncertainty<DIMENSIONS, PRECISION>>
TCoopPtr<THist<DIMENSIONS, PRECISION>>
MakeCoOwnedHist(std::array<TAxisConfig, DIMENSIONS> axes, STATISTICS statConfig = STATISTICS()) {
  THist<DIMENSIONS, PRECISION> hist(axes, statConfig);
  return MakeCoop(std::move(hist));
};


/// Adopt an external, stand-alone THistImpl. The THist will take ownership.
template <int DIMENSIONS, class PRECISION>
THist<DIMENSIONS, PRECISION>
HistFromImpl(std::unique_ptr<typename THist<DIMENSIONS, PRECISION>::ImplBase_t> pHistImpl) {
  THist<DIMENSIONS, PRECISION> ret;
  ret.fFillFunc = pHistImpl->GetFillFunc();
  std::swap(ret.fImpl, pHistImpl);
  return ret;
};


namespace Internal {
/**
 Generate THist::fImpl from THist constructor arguments.
 */
template<int DIMENSIONS, int IDIM, class PRECISION, class STATISTICS,
   class... PROCESSEDAXISCONFIG>
struct HistImplGen_t {
  /// Select the template argument for the next axis type, and "recurse" into
  /// HistImplGen_t for the next axis.
  template<TAxisConfig::EKind KIND>
  std::unique_ptr<THistImplBase<DIMENSIONS, PRECISION>>
  MakeNextAxis(const std::array<TAxisConfig, DIMENSIONS> &axes,
               const STATISTICS& statConfig,
               PROCESSEDAXISCONFIG... processedAxisArgs) {
    typename AxisConfigToType<KIND>::Axis_t nextAxis
       = AxisConfigToType<KIND>()(axes[IDIM]);
    return HistImplGen_t<DIMENSIONS, IDIM + 1, PRECISION, STATISTICS,
       PROCESSEDAXISCONFIG..., typename AxisConfigToType<KIND>::Axis_t>()
       (axes, statConfig, processedAxisArgs..., nextAxis);
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
  /// (`IDIM` == `DIMENSIONS`), all `axes` have been converted to
  /// `processedAxisArgs` and the THistImpl constructor can be invoked, passing
  /// the `processedAxisArgs`.
  std::unique_ptr<THistImplBase<DIMENSIONS, PRECISION>>
  operator()(const std::array<TAxisConfig, DIMENSIONS> &axes,
             const STATISTICS& statConfig,
             PROCESSEDAXISCONFIG... processedAxisArgs) {
    switch (axes[IDIM].GetKind()) {
      case TAxisConfig::kEquidistant:
        return MakeNextAxis<TAxisConfig::kEquidistant>(axes, statConfig,
                                                       processedAxisArgs...);
      case TAxisConfig::kGrow:
        return MakeNextAxis<TAxisConfig::kGrow>(axes, statConfig,
                                                processedAxisArgs...);
      case TAxisConfig::kIrregular:
        return MakeNextAxis<TAxisConfig::kIrregular>(axes, statConfig,
                                                     processedAxisArgs...);
      default:
        R__ERROR_HERE("HIST") << "Unhandled axis kind";
    }
    return nullptr;
  }
};

/// Generate THist::fImpl from constructor arguments; recursion end.
template<int DIMENSIONS, class PRECISION, class STATISTICS,
   class... PROCESSEDAXISCONFIG>
/// Create the histogram, now that all axis types and initializier objects are
/// determined.
struct HistImplGen_t<DIMENSIONS, DIMENSIONS, PRECISION, STATISTICS,
   PROCESSEDAXISCONFIG...> {
  std::unique_ptr<THistImplBase<DIMENSIONS, PRECISION>>
  operator()(const std::array<TAxisConfig, DIMENSIONS> &, const STATISTICS& statConfig,
             PROCESSEDAXISCONFIG... axisArgs) {
    using HistImplt_t
      = THistImpl<DIMENSIONS, PRECISION, STATISTICS, PROCESSEDAXISCONFIG...>;
    return std::make_unique<HistImplt_t>(statConfig, axisArgs...);
  }
};
} // namespace Internal


template<int DIMENSIONS, class PRECISION>
template<class STATISTICS /*= THistStatUncertainty<DIMENSIONS, PRECISION>*/>
THist<DIMENSIONS, PRECISION>::THist(std::array<TAxisConfig, DIMENSIONS> axes,
             STATISTICS statConfig /*= STATISTICS()*/):
  fImpl{std::move(
     Internal::HistImplGen_t<DIMENSIONS, 0, PRECISION, STATISTICS>()(axes,
                                                                   statConfig))},
  fFillFunc{} {
  fFillFunc = fImpl->GetFillFunc();
}

/// \name THist Typedefs
///\{ Convenience typedefs (ROOT6-compatible type names)

// Keep them as typedefs, to make sure old-style documentation tools can
// understand them.
typedef THist<1, double> TH1D;
typedef THist<1, float> TH1F;
typedef THist<1, char> TH1C;
typedef THist<1, int> TH1I;
typedef THist<1, int64_t> TH1LL;

typedef THist<2, double> TH2D;
typedef THist<2, float> TH2F;
typedef THist<2, char> TH2C;
typedef THist<2, int> TH2I;
typedef THist<2, int64_t> TH2LL;

typedef THist<3, double> TH3D;
typedef THist<3, float> TH3F;
typedef THist<3, char> TH3C;
typedef THist<3, int> TH3I;
typedef THist<3, int64_t> TH3LL;
///\}


template <int DIMENSION, class PRECISIONA, class PRECISIONB>
void Add(THist<DIMENSION, PRECISIONA>& to, THist<DIMENSION, PRECISIONB>& from) {
  using ImplTo_t = typename THist<DIMENSION, PRECISIONA>::ImplBase_t;
  using ImplFrom_t = typename THist<DIMENSION, PRECISIONB>::ImplBase_t;
  ImplTo_t* implTo = to.GetImpl();
  ImplFrom_t* implFrom = from.GetImpl();
  // TODO: move into THistImpl; the loop iteration should not go through virt interfaces!
  for (auto&& bin: from) {
    to.Fill(implFrom->GetBinCenter(*bin), implFrom->GetBinContent(*bin));
  }
};

template <int DIMENSION, class PRECISION>
std::unique_ptr<Internal::TDrawable>
GetDrawable(TCoopPtr<THist<DIMENSION, PRECISION>> hist,
            THistDrawOptions<DIMENSION> opts = {}) {
  return std::make_unique<Internal::THistDrawable<DIMENSION, PRECISION>>(hist, opts);
}

} // namespace ROOT

#endif
