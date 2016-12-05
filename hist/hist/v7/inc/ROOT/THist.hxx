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

#include "ROOT/RArrayView.hxx"
#include "ROOT/TAxis.hxx"
#include "ROOT/TDrawable.hxx"
#include "ROOT/THistBinIter.hxx"
#include "ROOT/THistDrawable.hxx"
#include "ROOT/THistImpl.hxx"
#include "ROOT/THistData.hxx"
#include <initializer_list>

namespace ROOT {
namespace Experimental {

// fwd declare for fwd declare for friend declaration in THist...
template<int DIMENSIONS, class PRECISION,
         template <int D_, class P_, template <class P__> class S_> class... STAT>
class THist;

// fwd declare for friend declaration in THist.
template<int DIMENSIONS, class PRECISION,
         template <int D_, class P_, template <class P__> class S_> class... STAT>
class THist<DIMENSIONS, PRECISION, STAT...>
  HistFromImpl(std::unique_ptr<typename THist<DIMENSIONS, PRECISION, STAT...>::ImplBase_t> pHistImpl);

/**
 \class THist
 Histogram class for histograms with `DIMENSIONS` dimensions, where each
 bin count is stored by a value of type `PRECISION`. STAT stores statistical
 data of the entries filled into the histogram (bin content, uncertainties etc).

 A histogram counts occurrences of values or n-dimensional combinations thereof.
 Contrary to for instance a `TTree`, a histogram combines adjacent values. The
 resolution of this combination is defined by the axis binning, see e.g.
 http://www.wikiwand.com/en/Histogram
 */

template<int DIMENSIONS, class PRECISION,
         template <int D_, class P_, template <class P__> class S_> class... STAT>
class THist {
public:
  /// The type of the `Detail::THistImplBase` of this histogram.
  using ImplBase_t
    = Detail::THistImplBase<Detail::THistData<DIMENSIONS, PRECISION,
                                              Detail::THistDataDefaultStorage, STAT...>>;
  /// The coordinates type: a `DIMENSIONS`-dimensional `std::array` of `double`.
  using CoordArray_t = typename ImplBase_t::CoordArray_t;
  /// The type of weights
  using Weight_t = PRECISION;
  /// Pointer type to `HistImpl_t::Fill`, for faster access.
  using FillFunc_t = typename ImplBase_t::FillFunc_t;
  /// Range.
  using AxisRange_t = typename ImplBase_t::AxisIterRange_t;

  using const_iterator = Detail::THistBinIter<ImplBase_t>;

  /// Number of dimensions of the coordinates
  static constexpr int GetNDim() noexcept { return DIMENSIONS; }

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
  explicit THist(std::array<TAxisConfig, DIMENSIONS> axes);

  /// Constructor overload taking the histogram title
  THist(std::string_view histTitle, std::array<TAxisConfig, DIMENSIONS> axes);

  /// Constructor overload that's only available for a 1-dimensional histogram.
  template <int ENABLEIF_NDIM = DIMENSIONS,
            class = typename std::enable_if<ENABLEIF_NDIM == 1>::type>
  explicit THist(const TAxisConfig &xaxis):
    THist(std::array<TAxisConfig, 1>{{xaxis}})
  { }

  /// Constructor overload that's only available for a 1-dimensional histogram,
  /// also passing the histogram title.
  template <int ENABLEIF_NDIM = DIMENSIONS,
            class = typename std::enable_if<ENABLEIF_NDIM == 1>::type>
  THist(std::string_view histTitle, const TAxisConfig &xaxis):
    THist(histTitle, std::array<TAxisConfig, 1>{{xaxis}})
  { }

  /// Constructor overload that's only available for a 2-dimensional histogram.
  template<int ENABLEIF_NDIM = DIMENSIONS,
           class = typename std::enable_if<ENABLEIF_NDIM == 2>::type>
  THist(const TAxisConfig &xaxis, const TAxisConfig &yaxis):
    THist(std::array<TAxisConfig, 2>{{xaxis, yaxis}})
  { }

  /// Constructor overload that's only available for a 2-dimensional histogram,
  /// also passing the histogram title.
  template<int ENABLEIF_NDIM = DIMENSIONS,
           class = typename std::enable_if<ENABLEIF_NDIM == 2>::type>
  THist(std::string_view histTitle, const TAxisConfig &xaxis, const TAxisConfig &yaxis):
    THist(histTitle, std::array<TAxisConfig, 2>{{xaxis, yaxis}})
  { }

  /// Constructor overload that's only available for a 3-dimensional histogram.
  template<int ENABLEIF_NDIM = DIMENSIONS,
           class = typename std::enable_if<ENABLEIF_NDIM == 3>::type>
  THist(const TAxisConfig &xaxis, const TAxisConfig &yaxis, const TAxisConfig &zaxis):
    THist(std::array<TAxisConfig, 3>{{xaxis, yaxis, zaxis}})
  { }

  /// Constructor overload that's only available for a 3-dimensional histogram,
  /// also passing the histogram title.
  template<int ENABLEIF_NDIM = DIMENSIONS,
           class = typename std::enable_if<ENABLEIF_NDIM == 3>::type>
  THist(std::string_view histTitle,
        const TAxisConfig &xaxis, const TAxisConfig &yaxis, const TAxisConfig &zaxis):
    THist(histTitle, std::array<TAxisConfig, 3>{{xaxis, yaxis, zaxis}})
  { }


  /// Access the ImplBase_t this THist points to.
  ImplBase_t *GetImpl() const noexcept { return fImpl.get(); }

  /// "Steal" the ImplBase_t this THist points to.
  std::unique_ptr<ImplBase_t>&& TakeImpl() noexcept { return std::move(fImpl); }

  /// Add `weight` to the bin containing coordinate `x`.
  void Fill(const CoordArray_t &x, Weight_t weight = (Weight_t) 1) noexcept { (fImpl.get()->*fFillFunc)(x, weight); }

  /// For each coordinate in `xN`, add `weightN[i]` to the bin at coordinate
  /// `xN[i]`. The sizes of `xN` and `weightN` must be the same. This is more
  /// efficient than many separate calls to `Fill()`.
  void FillN(const std::array_view <CoordArray_t> xN,
             const std::array_view <Weight_t> weightN) noexcept { fImpl->FillN(xN, weightN); }

  /// Convenience overload: `FillN()` with weight 1.
  void FillN(const std::array_view <CoordArray_t> xN) noexcept { fImpl->FillN(xN); }

  /// Get the number of entries this histogram was filled with.
  int64_t GetEntries() const noexcept { return fImpl->GetStat().GetEntries(); }

  /// Get the content of the bin at `x`.
  Weight_t GetBinContent(const CoordArray_t &x) const { return fImpl->GetBinContent(x); }

  /// Get the uncertainty on the content of the bin at `x`.
  double GetBinUncertainty(const CoordArray_t &x) const { return fImpl->GetBinUncertainty(x); }

  const_iterator begin() const { return const_iterator(*fImpl); }

  const_iterator end() const { return const_iterator(*fImpl, fImpl->GetNBins()); }

  /// Swap *this and other.
  ///
  /// Very efficient; swaps the `fImpl` pointers.
  void swap(THist<DIMENSIONS, PRECISION, STAT...> &other) noexcept {
    std::swap(fImpl, other.fImpl);
    std::swap(fFillFunc, other.fFillFunc);
  }

private:
  std::unique_ptr<ImplBase_t> fImpl; ///<  The actual histogram implementation
  FillFunc_t fFillFunc = nullptr;    ///<! Pinter to THistImpl::Fill() member function

  friend THist HistFromImpl<>(std::unique_ptr<ImplBase_t>);
};

/// THist with no STAT parameter uses THistStatContent by default.
template<int DIMENSIONS, class PRECISION>
class THist<DIMENSIONS, PRECISION>:
   public THist<DIMENSIONS, PRECISION, THistStatContent>
{
   using THist<DIMENSIONS, PRECISION, THistStatContent>::THist;
};


/// Swap two histograms.
///
/// Very efficient; swaps the `fImpl` pointers.
template<int DIMENSIONS, class PRECISION,
         template <int D_, class P_, template <class P__> class S_> class... STAT>
void swap(THist<DIMENSIONS, PRECISION, STAT...> &a,
          THist<DIMENSIONS, PRECISION, STAT...> &b) noexcept
{
  a.swap(b);
};


namespace Internal {
/**
 Generate THist::fImpl from THist constructor arguments.
 */
template<int NDIM, int IDIM, class DATA, class... PROCESSEDAXISCONFIG>
struct THistImplGen {
  /// Select the template argument for the next axis type, and "recurse" into
  /// THistImplGen for the next axis.
  template<TAxisConfig::EKind KIND>
  std::unique_ptr<Detail::THistImplBase<DATA>>
  MakeNextAxis(std::string_view title, const std::array<TAxisConfig, NDIM> &axes,
               PROCESSEDAXISCONFIG... processedAxisArgs)
  {
    using NextAxis_t = typename AxisConfigToType<KIND>::Axis_t;
    NextAxis_t nextAxis = AxisConfigToType<KIND>()(axes[IDIM]);
    using HistImpl_t = THistImplGen<NDIM, IDIM + 1, DATA, PROCESSEDAXISCONFIG..., NextAxis_t>;
    return HistImpl_t()(title, axes, processedAxisArgs..., nextAxis);
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
  operator()(std::string_view title, const std::array <TAxisConfig, NDIM> &axes,
             PROCESSEDAXISCONFIG... processedAxisArgs)
  {
    switch (axes[IDIM].GetKind()) {
      case TAxisConfig::kEquidistant:
        return MakeNextAxis<TAxisConfig::kEquidistant>(title, axes, processedAxisArgs...);
      case TAxisConfig::kGrow:
        return MakeNextAxis<TAxisConfig::kGrow>(title, axes, processedAxisArgs...);
      case TAxisConfig::kIrregular:
        return MakeNextAxis<TAxisConfig::kIrregular>(title, axes, processedAxisArgs...);
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
struct THistImplGen<NDIM, NDIM, DATA, PROCESSEDAXISCONFIG...> {
  using HistImplBase_t = ROOT::Experimental::Detail::THistImplBase<DATA>;
  std::unique_ptr<HistImplBase_t>
  operator()(std::string_view title, const std::array<TAxisConfig, DATA::GetNDim()> &, PROCESSEDAXISCONFIG... axisArgs)
  {
    using HistImplt_t = Detail::THistImpl<DATA, PROCESSEDAXISCONFIG...>;
    return std::make_unique<HistImplt_t>(title, axisArgs...);
  }
};
} // namespace Internal


template<int DIMENSIONS, class PRECISION,
         template <int D_, class P_, template <class P__> class S_> class... STAT>
THist<DIMENSIONS, PRECISION, STAT...>::THist(std::string_view title, std::array<TAxisConfig, DIMENSIONS> axes):
  fImpl{std::move(Internal::THistImplGen<THist::GetNDim(), 0,
        Detail::THistData<DIMENSIONS, PRECISION, Detail::THistDataDefaultStorage, STAT...>>()(title, axes))}
{
  fFillFunc = fImpl->GetFillFunc();
}


template<int DIMENSIONS, class PRECISION,
         template <int D_, class P_, template <class P__> class S_> class... STAT>
THist<DIMENSIONS, PRECISION, STAT...>::THist(std::array<TAxisConfig, DIMENSIONS> axes):
  THist("", axes) {}


/// Adopt an external, stand-alone THistImpl. The THist will take ownership.
template<int DIMENSIONS, class PRECISION,
         template <int D_, class P_, template <class P__> class S_> class... STAT>
THist<DIMENSIONS, PRECISION, STAT...>
HistFromImpl(std::unique_ptr<typename THist<DIMENSIONS, PRECISION, STAT...>::ImplBase_t> pHistImpl)
{
  THist<DIMENSIONS, PRECISION, STAT...> ret;
  ret.fFillFunc = pHistImpl->GetFillFunc();
  std::swap(ret.fImpl, pHistImpl);
  return ret;
};


/// \name THist Typedefs
///\{ Convenience typedefs (ROOT6-compatible type names)

// Keep them as typedefs, to make sure old-style documentation tools can understand them.
using TH1D  = THist<1, double, THistStatContent, THistStatUncertainty>;
using TH1F  = THist<1, float, THistStatContent, THistStatUncertainty>;
using TH1C  = THist<1, char, THistStatContent>;
using TH1I  = THist<1, int, THistStatContent>;
using TH1LL = THist<1, int64_t, THistStatContent>;

using TH2D  = THist<2, double, THistStatContent, THistStatUncertainty>;
using TH2F  = THist<2, float, THistStatContent, THistStatUncertainty>;
using TH2C  = THist<2, char, THistStatContent>;
using TH2I  = THist<2, int, THistStatContent>;
using TH2LL = THist<2, int64_t, THistStatContent>;

using TH3D  = THist<3, double, THistStatContent, THistStatUncertainty>;
using TH3F  = THist<3, float, THistStatContent, THistStatUncertainty>;
using TH3C  = THist<3, char, THistStatContent>;
using TH3I  = THist<3, int, THistStatContent>;
using TH3LL = THist<3, int64_t, THistStatContent>;
///\}


/// Add two histograms. This is the generic, inefficient version for now; it
/// assumes no matching axes.
template<int DIMENSIONS,
         class PRECISION_TO, class PRECISION_FROM,
         template <int D_, class P_, template <class P__> class S_> class... STAT_TO,
         template <int D_, class P_, template <class P__> class S_> class... STAT_FROM>
void Add(THist<DIMENSIONS, PRECISION_TO, STAT_TO...> &to,
         const THist<DIMENSIONS, PRECISION_FROM, STAT_FROM...> &from)
{
  auto toImpl = to.GetImpl();
  auto fillFuncTo = toImpl->GetFillFunc();
  using HistFrom_t = THist<DIMENSIONS, PRECISION_FROM, STAT_FROM...>;
  using FromCoord_t = typename HistFrom_t::CoordArray_t;
  using FromWeight_t = typename HistFrom_t::Weight_t;
  auto add = [fillFuncTo, toImpl](const FromCoord_t& x, FromWeight_t c)
  {
    (toImpl->*fillFuncTo)(x, c);
    // TODO: something nice with the uncertainty - depending on whether `to` cares
  };
  from.GetImpl()->ApplyXC(add);
}


/// Interface to graphics taking a unique_ptr<THist>.
template<int DIMENSIONS, class PRECISION,
         template <int D_, class P_, template <class P__> class S_> class... STAT>
std::unique_ptr <Internal::TDrawable>
GetDrawable(const std::shared_ptr<THist<DIMENSIONS, PRECISION, STAT...>>& hist,
            THistDrawOptions<DIMENSIONS> opts = {})
{
  return std::make_unique<Internal::THistDrawable<DIMENSIONS>>(hist, opts);
}

/// Interface to graphics taking a shared_ptr<THist>.
template<int DIMENSIONS, class PRECISION,
         template <int D_, class P_, template <class P__> class S_> class... STAT>
std::unique_ptr <Internal::TDrawable>
GetDrawable(std::unique_ptr<THist<DIMENSIONS, PRECISION, STAT...>>&& hist,
            THistDrawOptions<DIMENSIONS> opts = {})
{
  return std::make_unique<Internal::THistDrawable<DIMENSIONS>>(std::move(hist), opts);
}

} // namespace Experimental
} // namespace ROOT

#endif
