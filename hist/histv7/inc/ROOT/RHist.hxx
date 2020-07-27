/// \file ROOT/RHist.h
/// \ingroup Hist ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-03-23
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RHist
#define ROOT7_RHist

#include "ROOT/RSpan.hxx"
#include "ROOT/RAxis.hxx"
#include "ROOT/RHistBinIter.hxx"
#include "ROOT/RHistImpl.hxx"
#include "ROOT/RHistData.hxx"
#include <initializer_list>
#include <stdexcept>

#include "hist/hist/inc/HFitInterface.h"
#include "hist/hist/inc/TFitResult.h"
#include "hist/hist/inc/TFitResultPtr.h"
#include "hist/hist/inc/Foption.h"
#include "Fit/DataRange.h"
#include "Fit/BinData.h"
#include "Fit/DataOptions.h"
#include "Fit/FitConfig.h"
#include "Fit/Fitter.h"
#include "hist/hist/inc/TVirtualFitter.h"
#include "Fit/UnBinData.h"
#include "Fit/Chi2FCN.h"
#include "Fit/PoissonLikelihoodFCN.h"
#include "Math/MinimizerOptions.h"
#include "Math/Minimizer.h"
#include "math/mathcore/inc/TMath.h"
#include "Fit/FitExecutionPolicy.h"

#include "Math/WrappedTF1.h"
#include "Math/WrappedMultiTF1.h"
#include "Math/IParamFunction.h"


namespace ROOT {
namespace Experimental {

// fwd declare for fwd declare for friend declaration in RHist...
template <int DIMENSIONS, class PRECISION, template <int D_, class P_> class... STAT>
class RHist;

// fwd declare for friend declaration in RHist.
template <int DIMENSIONS, class PRECISION, template <int D_, class P_> class... STAT>
class RHist<DIMENSIONS, PRECISION, STAT...>
HistFromImpl(std::unique_ptr<typename RHist<DIMENSIONS, PRECISION, STAT...>::ImplBase_t> pHistImpl);

/**
 \class RHist
 Histogram class for histograms with `DIMENSIONS` dimensions, where each
 bin count is stored by a value of type `PRECISION`. STAT stores statistical
 data of the entries filled into the histogram (bin content, uncertainties etc).

 A histogram counts occurrences of values or n-dimensional combinations thereof.
 Contrary to for instance a `RTree`, a histogram combines adjacent values. The
 resolution of this combination is defined by the axis binning, see e.g.
 http://www.wikiwand.com/en/Histogram
 */

template <int DIMENSIONS, class PRECISION, template <int D_, class P_> class... STAT>
class RHist {
public:
   /// The type of the `Detail::RHistImplBase` of this histogram.
   using ImplBase_t =
      Detail::RHistImplBase<Detail::RHistData<DIMENSIONS, PRECISION, std::vector<PRECISION>, STAT...>>;
   /// The coordinates type: a `DIMENSIONS`-dimensional `std::array` of `double`.
   using CoordArray_t = typename ImplBase_t::CoordArray_t;
   /// The type of weights
   using Weight_t = PRECISION;
   /// Pointer type to `HistImpl_t::Fill`, for faster access.
   using FillFunc_t = typename ImplBase_t::FillFunc_t;
   /// Range.
   using AxisRange_t = typename ImplBase_t::AxisIterRange_t;

   using const_iterator = Detail::RHistBinIter<ImplBase_t>;

   /// Number of dimensions of the coordinates
   static constexpr int GetNDim() noexcept { return DIMENSIONS; }

   RHist() = default;
   RHist(RHist &&) = default;
   RHist(const RHist &other): fImpl(other.fImpl->Clone()), fFillFunc(other.fFillFunc)
   {}

   /// Create a histogram from an `array` of axes (`RAxisConfig`s). Example code:
   ///
   /// Construct a 1-dimensional histogram that can be filled with `floats`s.
   /// The axis has 10 bins between 0. and 1. The two outermost sets of curly
   /// braces are to reach the initialization of the `std::array` elements; the
   /// inner one is for the initialization of a `RAxisCoordinate`.
   ///
   ///     RHist<1,float> h1f({{ {10, 0., 1.} }});
   ///
   /// Construct a 2-dimensional histogram, with the first axis as before, and
   /// the second axis having non-uniform ("irregular") binning, where all bin-
   /// edges are specified. As this is itself an array it must be enclosed by
   /// double curlies.
   ///
   ///     RHist<2,int> h2i({{ {10, 0., 1.}, {{-1., 0., 1., 10., 100.}} }});
   explicit RHist(std::array<RAxisConfig, DIMENSIONS> axes);

   /// Constructor overload taking the histogram title.
   RHist(std::string_view histTitle, std::array<RAxisConfig, DIMENSIONS> axes);

   /// Constructor overload that's only available for a 1-dimensional histogram.
   template <int ENABLEIF_NDIM = DIMENSIONS, class = typename std::enable_if<ENABLEIF_NDIM == 1>::type>
   explicit RHist(const RAxisConfig &xaxis): RHist(std::array<RAxisConfig, 1>{{xaxis}})
   {}

   /// Constructor overload that's only available for a 1-dimensional histogram,
   /// also passing the histogram title.
   template <int ENABLEIF_NDIM = DIMENSIONS, class = typename std::enable_if<ENABLEIF_NDIM == 1>::type>
   RHist(std::string_view histTitle, const RAxisConfig &xaxis): RHist(histTitle, std::array<RAxisConfig, 1>{{xaxis}})
   {}

   /// Constructor overload that's only available for a 2-dimensional histogram.
   template <int ENABLEIF_NDIM = DIMENSIONS, class = typename std::enable_if<ENABLEIF_NDIM == 2>::type>
   RHist(const RAxisConfig &xaxis, const RAxisConfig &yaxis): RHist(std::array<RAxisConfig, 2>{{xaxis, yaxis}})
   {}

   /// Constructor overload that's only available for a 2-dimensional histogram,
   /// also passing the histogram title.
   template <int ENABLEIF_NDIM = DIMENSIONS, class = typename std::enable_if<ENABLEIF_NDIM == 2>::type>
   RHist(std::string_view histTitle, const RAxisConfig &xaxis, const RAxisConfig &yaxis)
      : RHist(histTitle, std::array<RAxisConfig, 2>{{xaxis, yaxis}})
   {}

   /// Constructor overload that's only available for a 3-dimensional histogram.
   template <int ENABLEIF_NDIM = DIMENSIONS, class = typename std::enable_if<ENABLEIF_NDIM == 3>::type>
   RHist(const RAxisConfig &xaxis, const RAxisConfig &yaxis, const RAxisConfig &zaxis)
      : RHist(std::array<RAxisConfig, 3>{{xaxis, yaxis, zaxis}})
   {}

   /// Constructor overload that's only available for a 3-dimensional histogram,
   /// also passing the histogram title.
   template <int ENABLEIF_NDIM = DIMENSIONS, class = typename std::enable_if<ENABLEIF_NDIM == 3>::type>
   RHist(std::string_view histTitle, const RAxisConfig &xaxis, const RAxisConfig &yaxis, const RAxisConfig &zaxis)
      : RHist(histTitle, std::array<RAxisConfig, 3>{{xaxis, yaxis, zaxis}})
   {}

   /// Access the ImplBase_t this RHist points to.
   ImplBase_t *GetImpl() const noexcept { return fImpl.get(); }

   /// "Steal" the ImplBase_t this RHist points to.
   std::unique_ptr<ImplBase_t> TakeImpl() && noexcept { return std::move(fImpl); }

   /// Add `weight` to the bin containing coordinate `x`.
   void Fill(const CoordArray_t &x, Weight_t weight = (Weight_t)1) noexcept { (fImpl.get()->*fFillFunc)(x, weight); }

   /// For each coordinate in `xN`, add `weightN[i]` to the bin at coordinate
   /// `xN[i]`. The sizes of `xN` and `weightN` must be the same. This is more
   /// efficient than many separate calls to `Fill()`.
   void FillN(const std::span<const CoordArray_t> xN, const std::span<const Weight_t> weightN) noexcept
   {
      fImpl->FillN(xN, weightN);
   }

   /// Convenience overload: `FillN()` with weight 1.
   void FillN(const std::span<const CoordArray_t> xN) noexcept { fImpl->FillN(xN); }

   /// Get the number of entries this histogram was filled with.
   int64_t GetEntries() const noexcept { return fImpl->GetStat().GetEntries(); }

   /// Get the content of the bin at `x`.
   Weight_t GetBinContent(const CoordArray_t &x) const { return fImpl->GetBinContent(x); }

   /// Get the uncertainty on the content of the bin at `x`.
   double GetBinUncertainty(const CoordArray_t &x) const { return fImpl->GetBinUncertainty(x); }

   const_iterator begin() const { return const_iterator(*fImpl); }

   const_iterator end() const { return const_iterator(*fImpl, fImpl->GetNBinsNoOver()); }

   /// Swap *this and other.
   ///
   /// Very efficient; swaps the `fImpl` pointers.
   void swap(RHist<DIMENSIONS, PRECISION, STAT...> &other) noexcept
   {
      std::swap(fImpl, other.fImpl);
      std::swap(fFillFunc, other.fFillFunc);
   }

private:
   std::unique_ptr<ImplBase_t> fImpl; ///< The actual histogram implementation.
   FillFunc_t fFillFunc = nullptr;    ///< Pointer to RHistImpl::Fill() member function.

   friend RHist HistFromImpl<>(std::unique_ptr<ImplBase_t>);
};

/// RHist with no STAT parameter uses RHistStatContent by default.
template <int DIMENSIONS, class PRECISION>
class RHist<DIMENSIONS, PRECISION>: public RHist<DIMENSIONS, PRECISION, RHistStatContent> {
   using RHist<DIMENSIONS, PRECISION, RHistStatContent>::RHist;
};

/// Swap two histograms.
///
/// Very efficient; swaps the `fImpl` pointers.
template <int DIMENSIONS, class PRECISION, template <int D_, class P_> class... STAT>
void swap(RHist<DIMENSIONS, PRECISION, STAT...> &a, RHist<DIMENSIONS, PRECISION, STAT...> &b) noexcept
{
   a.swap(b);
};

namespace Internal {
/**
 Generate RHist::fImpl from RHist constructor arguments.
 */
template <int NDIM, int IDIM, class DATA, class... PROCESSEDAXISCONFIG>
struct RHistImplGen {
   /// Select the template argument for the next axis type, and "recurse" into
   /// RHistImplGen for the next axis.
   template <RAxisConfig::EKind KIND>
   std::unique_ptr<Detail::RHistImplBase<DATA>>
   MakeNextAxis(std::string_view title, const std::array<RAxisConfig, NDIM> &axes,
                PROCESSEDAXISCONFIG... processedAxisArgs)
   {
      using NextAxis_t = typename AxisConfigToType<KIND>::Axis_t;
      NextAxis_t nextAxis = AxisConfigToType<KIND>()(axes[IDIM]);
      using HistImpl_t = RHistImplGen<NDIM, IDIM + 1, DATA, PROCESSEDAXISCONFIG..., NextAxis_t>;
      return HistImpl_t()(title, axes, processedAxisArgs..., nextAxis);
   }

   /// Make a RHistImpl-derived object reflecting the RAxisConfig array.
   ///
   /// Delegate to the appropriate MakeNextAxis instantiation, depending on the
   /// axis type selected in the RAxisConfig.
   /// \param axes - `RAxisConfig` objects describing the axis of the resulting
   ///   RHistImpl.
   /// \param statConfig - the statConfig parameter to be passed to the RHistImpl
   /// \param processedAxisArgs - the RAxisBase-derived axis objects describing the
   ///   axes of the resulting RHistImpl. There are `IDIM` of those; in the end
   /// (`IDIM` == `GetNDim()`), all `axes` have been converted to
   /// `processedAxisArgs` and the RHistImpl constructor can be invoked, passing
   /// the `processedAxisArgs`.
   std::unique_ptr<Detail::RHistImplBase<DATA>> operator()(std::string_view title,
                                                           const std::array<RAxisConfig, NDIM> &axes,
                                                           PROCESSEDAXISCONFIG... processedAxisArgs)
   {
      switch (axes[IDIM].GetKind()) {
      case RAxisConfig::kEquidistant: return MakeNextAxis<RAxisConfig::kEquidistant>(title, axes, processedAxisArgs...);
      case RAxisConfig::kGrow: return MakeNextAxis<RAxisConfig::kGrow>(title, axes, processedAxisArgs...);
      case RAxisConfig::kIrregular: return MakeNextAxis<RAxisConfig::kIrregular>(title, axes, processedAxisArgs...);
      default: R__ERROR_HERE("HIST") << "Unhandled axis kind";
      }
      return nullptr;
   }
};

/// Generate RHist::fImpl from constructor arguments; recursion end.
template <int NDIM, class DATA, class... PROCESSEDAXISCONFIG>
/// Create the histogram, now that all axis types and initializer objects are
/// determined.
struct RHistImplGen<NDIM, NDIM, DATA, PROCESSEDAXISCONFIG...> {
   using HistImplBase_t = ROOT::Experimental::Detail::RHistImplBase<DATA>;
   std::unique_ptr<HistImplBase_t>
   operator()(std::string_view title, const std::array<RAxisConfig, DATA::GetNDim()> &, PROCESSEDAXISCONFIG... axisArgs)
   {
      using HistImplt_t = Detail::RHistImpl<DATA, PROCESSEDAXISCONFIG...>;
      return std::make_unique<HistImplt_t>(title, axisArgs...);
   }
};
} // namespace Internal

template <int DIMENSIONS, class PRECISION, template <int D_, class P_> class... STAT>
RHist<DIMENSIONS, PRECISION, STAT...>::RHist(std::string_view title, std::array<RAxisConfig, DIMENSIONS> axes)
   : fImpl{std::move(
        Internal::RHistImplGen<RHist::GetNDim(), 0,
                               Detail::RHistData<DIMENSIONS, PRECISION, std::vector<PRECISION>, STAT...>>()(
           title, axes))}
{
   fFillFunc = fImpl->GetFillFunc();
}

template <int DIMENSIONS, class PRECISION, template <int D_, class P_> class... STAT>
RHist<DIMENSIONS, PRECISION, STAT...>::RHist(std::array<RAxisConfig, DIMENSIONS> axes): RHist("", axes)
{}

/// Adopt an external, stand-alone RHistImpl. The RHist will take ownership.
template <int DIMENSIONS, class PRECISION, template <int D_, class P_> class... STAT>
RHist<DIMENSIONS, PRECISION, STAT...>
HistFromImpl(std::unique_ptr<typename RHist<DIMENSIONS, PRECISION, STAT...>::ImplBase_t> pHistImpl)
{
   RHist<DIMENSIONS, PRECISION, STAT...> ret;
   ret.fFillFunc = pHistImpl->GetFillFunc();
   std::swap(ret.fImpl, pHistImpl);
   return ret;
};

/// \name RHist Typedefs
///\{ Convenience typedefs (ROOT6-compatible type names)

// Keep them as typedefs, to make sure old-style documentation tools can understand them.
using RH1D = RHist<1, double, RHistStatContent, RHistStatUncertainty>;
using RH1F = RHist<1, float, RHistStatContent, RHistStatUncertainty>;
using RH1C = RHist<1, char, RHistStatContent>;
using RH1I = RHist<1, int, RHistStatContent>;
using RH1LL = RHist<1, int64_t, RHistStatContent>;

using RH2D = RHist<2, double, RHistStatContent, RHistStatUncertainty>;
using RH2F = RHist<2, float, RHistStatContent, RHistStatUncertainty>;
using RH2C = RHist<2, char, RHistStatContent>;
using RH2I = RHist<2, int, RHistStatContent>;
using RH2LL = RHist<2, int64_t, RHistStatContent>;

using RH3D = RHist<3, double, RHistStatContent, RHistStatUncertainty>;
using RH3F = RHist<3, float, RHistStatContent, RHistStatUncertainty>;
using RH3C = RHist<3, char, RHistStatContent>;
using RH3I = RHist<3, int, RHistStatContent>;
using RH3LL = RHist<3, int64_t, RHistStatContent>;
///\}

/// Add two histograms.
///
/// This operation may currently only be performed if the two histograms have
/// the same axis configuration, use the same precision, and if `from` records
/// at least the same statistics as `to` (recording more stats is fine).
///
/// Adding histograms with incompatible axis binning will be reported at runtime
/// with an `std::runtime_error`. Insufficient statistics in the source
/// histogram will be detected at compile-time and result in a compiler error.
///
/// In the future, we may either adopt a more relaxed definition of histogram
/// addition or provide a mechanism to convert from one histogram type to
/// another. We currently favor the latter path.
template <int DIMENSIONS, class PRECISION,
          template <int D_, class P_> class... STAT_TO,
          template <int D_, class P_> class... STAT_FROM>
void Add(RHist<DIMENSIONS, PRECISION, STAT_TO...> &to, const RHist<DIMENSIONS, PRECISION, STAT_FROM...> &from)
{
   // Enforce "same axis configuration" policy.
   auto& toImpl = *to.GetImpl();
   const auto& fromImpl = *from.GetImpl();
   for (int dim = 0; dim < DIMENSIONS; ++dim) {
      if (!toImpl.GetAxis(dim).HasSameBinningAs(fromImpl.GetAxis(dim))) {
         throw std::runtime_error("Attempted to add RHists with incompatible axis binning");
      }
   }

   // Now that we know that the two axes have the same binning, we can just add
   // the statistics directly.
   toImpl.GetStat().Add(fromImpl.GetStat());
}

/// For now, only for dim <= 3 (due to the use of functions restricting dimension like in DataRange) 
template <int DIMENSIONS, class PRECISION, template <int D_, class P_> class... STAT>
TFitResultPtr Fit(RHist<DIMENSIONS, PRECISION, STAT...> & hist, TF1 *f1, const DataOptions & fitOption, const FitConfig & fitConfig)
//TFitResultPtr Fit(RHist<DIMENSIONS, PRECISION, STAT...> & hist, IParamFunction & func, const DataOptions & fitOption, const FitConfig & fitConfig)
//TFitResultPtr TH1::Fit(TF1 *f1 ,Option_t *option ,Option_t *goption, Double_t xxmin, Double_t xxmax)
{
   // Make sure function and histogram are compatible for fitting
   int ndim = hist.GetNDim();
   if (ndim == 0 || ndim > 3)
      // Arbitrary error return
      return -1;
   int checkResult = HFit::CheckFitFunction(f1, ndim);
   if (checkResult != 0)
      return checkResult;

   // If function dimension is less than hist dimension, then integral option is not possible
   if (f1->GetNdim() < ndim ) {
      if (fitOption.fIntegral) Info("Fit","Ignore Integral option. Model function dimension is less than the data object dimension");
      fitOption.fIntegral = 0;
   }

   // If specified, use range of function when fitting
   ROOT::Fit::DataRange & range = new ROOT::Fit::DataRange(ndim);
   if (fitOption.fUseRange) {
      HFit::GetFunctionRange(*f1,range);
   }

   Int_t special = f1->GetNumber();
   Bool_t linear = f1->IsLinear();
   Int_t npar = f1->GetNpar();
   // If polynomial function, make linear
   if (special == 299 + npar)  linear = kTRUE;
   // If option "integral" true, make non linear
   if (fitOption.fIntegral)
      linear = kFALSE;

   // Create an empty TFitResult, result of the fitting
   std::shared_ptr<TFitResult> tFitResult(new TFitResult());
   // Create the fitter from an empty fit result
   std::shared_ptr<ROOT::Fit::Fitter> fitter(new ROOT::Fit::Fitter(std::static_pointer_cast<ROOT::Fit::FitResult>(tFitResult)));
   // Set config options for fitter
   ROOT::Fit::FitConfig & fitterConfig = fitter->Config();
   fitterConfig = fitConfig;

   // Option special cases
   if (fitOption.fExpErrors) fitOption.fUseEmpty = true;  // use empty bins in log-likelihood fits
   if (special == 300) fitOption.fCoordErrors = false; // no need to use coordinate errors in a pol0 fit
   if (!fitOption.fErrors1) fitOption.fUseEmpty = true; // use empty bins with weight=1

   // Fill data for fitting
   std::shared_ptr<ROOT::Fit::BinData> fitData(new ROOT::Fit::BinData(fitOption,range));
   // TODO: check the different errors/uncertainty wanted for implementation
   int nPoints = hist.GetImpl()->GetNBins();

   bool useRange = (range.Size(0) > 0);
   double xmin = 0, ymin = 0, zmin = 0;
   double xmax = 0, ymax = 0, zmax = 0;
   switch (ndim) {
      case 1:
         range.GetRange(xmin,xmax);
         break;
      case 2:
         range.GetRange(xmin,xmax,ymin,ymax);
         break;
      case 3:
         range.GetRange(xmin,xmax,ymin,ymax,zmin,zmax);
      default:
         break;
   }

   ROOT::Fit::BinData::ErrorType errorType = ROOT::Fit::BinData::kValueError;
   if (fitOption.fErrors1) {
      errorType =  ROOT::Fit::BinData::kNoError;
   }
   fitData.Initialize(nPoints, ndim, errorType);
   
   for (auto bin: hist) {

      if (useRange && (bin.GetFrom()[0] < xmin || bin.GetTo()[0] > xmax || 
                        bin.GetFrom()[1] < ymin || bin.GetTo()[1] > ymax ||
                        bin.GetFrom()[2] < zmin || bin.GetTo()[2] > zmax ))
         continue;

      if (f1) {
         TF1::RejectPoint(false);
         (*f1)(bin);
         if (TF1::RejectedPoint())
            continue;
      }

      if (fitOption.fErrors1) {
         fitData.Add(bin.GetCenter(), bin.GetContent());
      }
      else if (errorType == ROOT::Fit::BinData::kValueError) {
         if (!ROOT::Fit::HFitInterface::AdjustError(fitOption, bin.GetUncertainty()))
            continue;
         fitData.Add(bin.GetCenter(), bin.GetContent(), bin.GetUncertainty());
      }
   }
   
   if (fitData->Size() == 0) {
      Warning("Fit","Fit data is empty ");
      return -1;
   }

   ROOT::Math::MinimizerOptions * fMinimizerOpts = fitConfig->MinimizerOptions();

   // Switch off linear fitting in case data has coordinate errors and the option is set
   if (fitData->GetErrorType() == ROOT::Fit::BinData::kCoordError && fitData->Opt().fCoordErrors ) linear = false;
   // Linear fit cannot be done also in case of asymmetric errors
   if (fitData->GetErrorType() == ROOT::Fit::BinData::kAsymError && fitData->Opt().fAsymErrors ) linear = false;

   // TShis functions use the TVirtualFitter
   if (special != 0 && !linear) {
      if      (special == 100)      ROOT::Fit::InitGaus  (*fitData,f1); // gaussian
      else if (special == 110 || special == 112)   ROOT::Fit::Init2DGaus(*fitData,f1); // 2D gaussians ( xygaus or bigaus)
      else if (special == 400)      ROOT::Fit::InitGaus  (*fitData,f1); // landau (use the same)
      else if (special == 410)      ROOT::Fit::Init2DGaus(*fitData,f1); // 2D landau (use the same)
      else if (special == 200)      ROOT::Fit::InitExpo  (*fitData, f1); // exponential

   }

   // Set the fit function
   if (linear)
      fitter->SetFunction(ROOT::Math::WrappedMultiTF1(*f1));
   else
      fitter->SetFunction(static_cast<const ROOT::Math::IParamMultiFunction &>(ROOT::Math::WrappedMultiTF1(*f1)));
   // Create the wrapped TF1 function to transform into IParamFunction
   //ROOT::Math::WrappedTF1 wrappedTF1(*f1);
   // Create the IParamFunction
   //ROOT::Math::IParamFunction & paramFunction = wrappedTF1;

   // Error normalization in case of zero error in the data
   if (fitData->GetErrorType() == ROOT::Fit::BinData::kNoError) fitterConfig.SetNormErrors(true);
   // Error normalization also in case of weights = 1
   if (fitData->Opt().fErrors1)  fitterConfig.SetNormErrors(true);
   // Normalize errors also in case you are fitting a Ndim histo with a N-1 function
   if (int(fitData->NDim())  == ndim -1 ) fitterConfig.SetNormErrors(true);

   // Parameter settings and transfer of the parameters values, names and limits from the functions
   // are done automatically in Fitter.cxx
   for (int i = 0; i < npar; ++i) {
      ROOT::Fit::ParameterSettings & parSettings = fitterConfig.ParSettings(i);

      // check limits
      double plow,pup;
      f1->GetParLimits(i,plow,pup);
      if (plow*pup != 0 && plow >= pup) { // this is a limitation - cannot fix a parameter to zero value
         parSettings.Fix();
      }
      else if (plow < pup ) {
         if (!TMath::Finite(pup) && TMath::Finite(plow) )
            parSettings.SetLowerLimit(plow);
         else if (!TMath::Finite(plow) && TMath::Finite(pup) )
            parSettings.SetUpperLimit(pup);
         else
            parSettings.SetLimits(plow,pup);
      }

      // set the parameter step size (by default are set to 0.3 of value)
      // if function provides meaningful error values
      double err = f1->GetParError(i);
      if ( err > 0)
         parSettings.SetStepSize(err);
      else if (plow < pup && TMath::Finite(plow) && TMath::Finite(pup) ) { // in case of limits improve step sizes
         double step = 0.1 * (pup - plow);
         // check if value is not too close to limit otherwise trim value
         if (  parSettings.Value() < pup && pup - parSettings.Value() < 2 * step  )
            step = (pup - parSettings.Value() ) / 2;
         else if ( parSettings.Value() > plow && parSettings.Value() - plow < 2 * step )
            step = (parSettings.Value() - plow ) / 2;

         parSettings.SetStepSize(step);
      }
   }

   // Set all default minimizer options (tolerance, max iterations, etc..)
   fitterConfig.SetMinimizerOptions(fMinimizerOpts);

   // Specific minimizer options depending on minimizer
   if (linear) {
      fitterConfig.SetMinimizer("Linear","");
   }

   // Run fitting
   bool fitDone = false;

   fitDone = fitter->Fit(fitData, ROOT::Fit::ExecutionPolicy::kSerial);

   checkResult |= !fitDone;

   const ROOT::Fit::FitResult & fitResult = fitter->Result();

   // Set directly the fit result in TF1
   checkResult = fitResult.Status();
   if (!fitResult.IsEmpty()) {
      f1->SetChisquare(fitResult.Chi2());
      f1->SetNDF(fitResult.Ndf());
      f1->SetNumberFitPoints(fitData->Size());

      assert((Int_t)fitResult.Parameters().size() >= f1->GetNpar());
      f1->SetParameters( const_cast<double*>(&(fitResult.Parameters().front())));
      if (int(fitResult.Errors().size()) >= f1->GetNpar())
         f1->SetParErrors( &(fitResult.Errors().front()));
   }

   // Print the result of the fitting
   if (fitter->GetMinimizer() && fitConfig.MinimizerType() == "Minuit" &&
            !fitConfig.NormalizeErrors()) {
         fitter->GetMinimizer()->PrintResults();
   }
   else {
      fitResult.PrintCovMatrix(std::cout);
      fitResult.Print(std::cout);
   }

   return TFitResultPtr(checkResult);

   // ROOT::Fit::BinData d;
   // ROOT::Fit::FillData(d,h1,func);

   // printData(d);

   // double p[3] = {100,0,3.};
   // f.SetParameters(p);

   // // create the fitter

   // ROOT::Fit::Fitter fitter;

   //bool ret = fitter.Fit(fitData, paramFunction);

   // ROOT::Fit::DataOptions opt;
   // opt.fUseEmpty = true;
   // ROOT::Fit::BinData dl(opt);
   // ROOT::Fit::FillData(dl,h1,func);
}


} // namespace Experimental
} // namespace ROOT

#endif
