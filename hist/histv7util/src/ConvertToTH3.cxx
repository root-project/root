/// \file
/// \warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
/// Feedback is welcome!

#include <ROOT/Hist/ConversionUtils.hxx>
#include <ROOT/Hist/ConvertToTH3.hxx>
#include <ROOT/RBinIndex.hxx>
#include <ROOT/RHist.hxx>
#include <ROOT/RHistEngine.hxx>

#include <TH3.h>

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
std::unique_ptr<Hist> ConvertToTH3Impl(const RHistEngine<T> &engine)
{
   if (engine.GetNDimensions() != 3) {
      throw std::invalid_argument("TH3 requires three dimensions");
   }

   auto ret = std::make_unique<Hist>();
   ret->SetDirectory(nullptr);

   const auto &axis0 = engine.GetAxes()[0];
   ROOT::Experimental::Hist::Internal::ConvertAxis(*ret->GetXaxis(), axis0);
   const auto &axis1 = engine.GetAxes()[1];
   ROOT::Experimental::Hist::Internal::ConvertAxis(*ret->GetYaxis(), axis1);
   const auto &axis2 = engine.GetAxes()[2];
   ROOT::Experimental::Hist::Internal::ConvertAxis(*ret->GetZaxis(), axis2);
   ret->SetBinsLength();

   Double_t *sumw2 = nullptr;
   auto copyBinContent = [&ret, &engine, &sumw2](Int_t i, RBinIndex index0, RBinIndex index1, RBinIndex index2) {
      if constexpr (std::is_same_v<T, RBinWithError>) {
         if (sumw2 == nullptr) {
            ret->Sumw2();
            sumw2 = ret->GetSumw2()->GetArray();
         }
         const RBinWithError &c = engine.GetBinContent(index0, index1, index2);
         ret->GetArray()[i] = c.fSum;
         sumw2[i] = c.fSum2;
      } else {
         (void)sumw2;
         ret->GetArray()[i] = engine.GetBinContent(index0, index1, index2);
      }
   };

   // Copy the bin contents, accounting for TH3 numbering conventions.
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
         Int_t i1 = 0;
         if (index1.IsUnderflow()) {
            i1 = 0;
         } else if (index1.IsOverflow()) {
            i1 = axis1.GetNNormalBins() + 1;
         } else {
            assert(index1.IsNormal());
            i1 = index1.GetIndex() + 1;
         }

         Int_t n1 = ret->GetYaxis()->GetNbins() + 2;
         for (auto index2 : axis2.GetFullRange()) {
            Int_t i2 = 0;
            if (index2.IsUnderflow()) {
               i2 = 0;
            } else if (index2.IsOverflow()) {
               i2 = axis2.GetNNormalBins() + 1;
            } else {
               assert(index2.IsNormal());
               i2 = index2.GetIndex() + 1;
            }
            copyBinContent(i0 + n0 * (i1 + n1 * i2), index0, index1, index2);
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

   Double_t hStats[11] = {
      stats.GetSumW(),
      stats.GetSumW2(),
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
   // We do not have sumwxy for hStats[6]
   if (stats.IsEnabled(2)) {
      hStats[7] = stats.GetDimensionStats(2).fSumWX;
      hStats[8] = stats.GetDimensionStats(2).fSumWX2;
   }
   // We do not have sumwxz or sumwyz for hStats[9] and hStats[10]
   h.PutStats(hStats);
}
} // namespace

namespace ROOT {
namespace Experimental {
namespace Hist {

std::unique_ptr<TH3C> ConvertToTH3C(const RHistEngine<char> &engine)
{
   return ConvertToTH3Impl<TH3C>(engine);
}

std::unique_ptr<TH3S> ConvertToTH3S(const RHistEngine<short> &engine)
{
   return ConvertToTH3Impl<TH3S>(engine);
}

std::unique_ptr<TH3I> ConvertToTH3I(const RHistEngine<int> &engine)
{
   return ConvertToTH3Impl<TH3I>(engine);
}

std::unique_ptr<TH3L> ConvertToTH3L(const RHistEngine<long> &engine)
{
   return ConvertToTH3Impl<TH3L>(engine);
}

std::unique_ptr<TH3L> ConvertToTH3L(const RHistEngine<long long> &engine)
{
   return ConvertToTH3Impl<TH3L>(engine);
}

std::unique_ptr<TH3F> ConvertToTH3F(const RHistEngine<float> &engine)
{
   return ConvertToTH3Impl<TH3F>(engine);
}

std::unique_ptr<TH3D> ConvertToTH3D(const RHistEngine<double> &engine)
{
   return ConvertToTH3Impl<TH3D>(engine);
}

std::unique_ptr<TH3D> ConvertToTH3D(const RHistEngine<RBinWithError> &engine)
{
   return ConvertToTH3Impl<TH3D>(engine);
}

std::unique_ptr<TH3C> ConvertToTH3C(const RHist<char> &hist)
{
   auto ret = ConvertToTH3C(hist.GetEngine());
   ConvertGlobalStatistics(*ret, hist.GetStats());
   return ret;
}

std::unique_ptr<TH3S> ConvertToTH3S(const RHist<short> &hist)
{
   auto ret = ConvertToTH3S(hist.GetEngine());
   ConvertGlobalStatistics(*ret, hist.GetStats());
   return ret;
}

std::unique_ptr<TH3I> ConvertToTH3I(const RHist<int> &hist)
{
   auto ret = ConvertToTH3I(hist.GetEngine());
   ConvertGlobalStatistics(*ret, hist.GetStats());
   return ret;
}

std::unique_ptr<TH3L> ConvertToTH3L(const RHist<long> &hist)
{
   auto ret = ConvertToTH3L(hist.GetEngine());
   ConvertGlobalStatistics(*ret, hist.GetStats());
   return ret;
}

std::unique_ptr<TH3L> ConvertToTH3L(const RHist<long long> &hist)
{
   auto ret = ConvertToTH3L(hist.GetEngine());
   ConvertGlobalStatistics(*ret, hist.GetStats());
   return ret;
}

std::unique_ptr<TH3F> ConvertToTH3F(const RHist<float> &hist)
{
   auto ret = ConvertToTH3F(hist.GetEngine());
   ConvertGlobalStatistics(*ret, hist.GetStats());
   return ret;
}

std::unique_ptr<TH3D> ConvertToTH3D(const RHist<double> &hist)
{
   auto ret = ConvertToTH3D(hist.GetEngine());
   ConvertGlobalStatistics(*ret, hist.GetStats());
   return ret;
}

std::unique_ptr<TH3D> ConvertToTH3D(const RHist<RBinWithError> &hist)
{
   auto ret = ConvertToTH3D(hist.GetEngine());
   ConvertGlobalStatistics(*ret, hist.GetStats());
   return ret;
}

} // namespace Hist
} // namespace Experimental
} // namespace ROOT
