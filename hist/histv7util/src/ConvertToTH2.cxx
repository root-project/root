/// \file
/// \warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
/// Feedback is welcome!

#include <ROOT/Hist/ConversionUtils.hxx>
#include <ROOT/Hist/ConvertToTH2.hxx>
#include <ROOT/RBinIndex.hxx>
#include <ROOT/RHist.hxx>
#include <ROOT/RHistEngine.hxx>

#include <TH2.h>

#include <cassert>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <variant>
#include <vector>

using namespace ROOT::Experimental;

namespace {
template <typename Hist, typename T>
std::unique_ptr<Hist> ConvertToTH2Impl(const RHistEngine<T> &engine)
{
   if (engine.GetNDimensions() != 2) {
      throw std::invalid_argument("TH2 requires two dimensions");
   }

   auto ret = std::make_unique<Hist>();
   ret->SetDirectory(nullptr);

   const auto &axis0 = engine.GetAxes()[0];
   ROOT::Experimental::Hist::Internal::ConvertAxis(*ret->GetXaxis(), axis0);
   const auto &axis1 = engine.GetAxes()[1];
   ROOT::Experimental::Hist::Internal::ConvertAxis(*ret->GetYaxis(), axis1);
   ret->SetBinsLength();

   Double_t *sumw2 = nullptr;
   auto copyBinContent = [&ret, &engine, &sumw2](Int_t i, RBinIndex index0, RBinIndex index1) {
      if constexpr (std::is_same_v<T, RBinWithError>) {
         if (sumw2 == nullptr) {
            ret->Sumw2();
            sumw2 = ret->GetSumw2()->GetArray();
         }
         const RBinWithError &c = engine.GetBinContent(index0, index1);
         ret->GetArray()[i] = c.fSum;
         sumw2[i] = c.fSum2;
      } else {
         (void)sumw2;
         ret->GetArray()[i] = engine.GetBinContent(index0, index1);
      }
   };

   // Copy the bin contents, accounting for TH2 numbering conventions.
   for (auto index0 : axis0.GetFullRange()) {
      Int_t i0 = 0;
      if (index0.IsUnderflow()) {
         i0 = 0;
      } else if (index0.IsOverflow()) {
         i0 = axis0.GetNNormalBins() + 1;
      } else {
         assert(index0.IsNormal());
         i0 = index0.GetIndex() + 1;
      }
      Int_t n0 = ret->GetXaxis()->GetNbins() + 2;

      for (auto index1 : axis1.GetFullRange()) {
         if (index1.IsUnderflow()) {
            copyBinContent(i0, index0, index1);
         } else if (index1.IsOverflow()) {
            copyBinContent(i0 + n0 * (axis1.GetNNormalBins() + 1), index0, index1);
         } else {
            assert(index1.IsNormal());
            copyBinContent(i0 + n0 * (index1.GetIndex() + 1), index0, index1);
         }
      }
   }

   return ret;
}

template <typename Hist>
void ConvertGlobalStatistics(Hist &h, const RHistStats &stats)
{
   if (stats.IsTainted()) {
      return;
   }

   h.SetEntries(stats.GetNEntries());

   Double_t hStats[7] = {
      stats.GetSumW(),
      stats.GetSumW2(),
      0,
      0,
      0,
      0,
      // We do not have sumwxy
      0,
   };
   if (stats.IsEnabled(0)) {
      hStats[2] = stats.GetDimensionStats(0).fSumWX;
      hStats[3] = stats.GetDimensionStats(0).fSumWX2;
   }
   if (stats.IsEnabled(1)) {
      hStats[4] = stats.GetDimensionStats(1).fSumWX;
      hStats[5] = stats.GetDimensionStats(1).fSumWX2;
   }
   h.PutStats(hStats);
}
} // namespace

namespace ROOT {
namespace Experimental {
namespace Hist {

std::unique_ptr<TH2C> ConvertToTH2C(const RHistEngine<char> &engine)
{
   return ConvertToTH2Impl<TH2C>(engine);
}

std::unique_ptr<TH2S> ConvertToTH2S(const RHistEngine<short> &engine)
{
   return ConvertToTH2Impl<TH2S>(engine);
}

std::unique_ptr<TH2I> ConvertToTH2I(const RHistEngine<int> &engine)
{
   return ConvertToTH2Impl<TH2I>(engine);
}

std::unique_ptr<TH2L> ConvertToTH2L(const RHistEngine<long> &engine)
{
   return ConvertToTH2Impl<TH2L>(engine);
}

std::unique_ptr<TH2L> ConvertToTH2L(const RHistEngine<long long> &engine)
{
   return ConvertToTH2Impl<TH2L>(engine);
}

std::unique_ptr<TH2F> ConvertToTH2F(const RHistEngine<float> &engine)
{
   return ConvertToTH2Impl<TH2F>(engine);
}

std::unique_ptr<TH2D> ConvertToTH2D(const RHistEngine<double> &engine)
{
   return ConvertToTH2Impl<TH2D>(engine);
}

std::unique_ptr<TH2D> ConvertToTH2D(const RHistEngine<RBinWithError> &engine)
{
   return ConvertToTH2Impl<TH2D>(engine);
}

std::unique_ptr<TH2C> ConvertToTH2C(const RHist<char> &hist)
{
   auto ret = ConvertToTH2C(hist.GetEngine());
   ConvertGlobalStatistics(*ret, hist.GetStats());
   return ret;
}

std::unique_ptr<TH2S> ConvertToTH2S(const RHist<short> &hist)
{
   auto ret = ConvertToTH2S(hist.GetEngine());
   ConvertGlobalStatistics(*ret, hist.GetStats());
   return ret;
}

std::unique_ptr<TH2I> ConvertToTH2I(const RHist<int> &hist)
{
   auto ret = ConvertToTH2I(hist.GetEngine());
   ConvertGlobalStatistics(*ret, hist.GetStats());
   return ret;
}

std::unique_ptr<TH2L> ConvertToTH2L(const RHist<long> &hist)
{
   auto ret = ConvertToTH2L(hist.GetEngine());
   ConvertGlobalStatistics(*ret, hist.GetStats());
   return ret;
}

std::unique_ptr<TH2L> ConvertToTH2L(const RHist<long long> &hist)
{
   auto ret = ConvertToTH2L(hist.GetEngine());
   ConvertGlobalStatistics(*ret, hist.GetStats());
   return ret;
}

std::unique_ptr<TH2F> ConvertToTH2F(const RHist<float> &hist)
{
   auto ret = ConvertToTH2F(hist.GetEngine());
   ConvertGlobalStatistics(*ret, hist.GetStats());
   return ret;
}

std::unique_ptr<TH2D> ConvertToTH2D(const RHist<double> &hist)
{
   auto ret = ConvertToTH2D(hist.GetEngine());
   ConvertGlobalStatistics(*ret, hist.GetStats());
   return ret;
}

std::unique_ptr<TH2D> ConvertToTH2D(const RHist<RBinWithError> &hist)
{
   auto ret = ConvertToTH2D(hist.GetEngine());
   ConvertGlobalStatistics(*ret, hist.GetStats());
   return ret;
}

} // namespace Hist
} // namespace Experimental
} // namespace ROOT
