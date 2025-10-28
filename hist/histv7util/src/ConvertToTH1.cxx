/// \file
/// \warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
/// Feedback is welcome!

#include <ROOT/Hist/ConvertToTH1.hxx>
#include <ROOT/RBinIndex.hxx>
#include <ROOT/RHist.hxx>
#include <ROOT/RHistEngine.hxx>

#include <TH1.h>

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
std::unique_ptr<Hist> ConvertToTH1Impl(const RHistEngine<T> &engine)
{
   if (engine.GetNDimensions() != 1) {
      throw std::invalid_argument("TH1 requires one dimension");
   }

   std::unique_ptr<Hist> ret;
   Double_t *sumw2 = nullptr;
   auto copyBinContent = [&ret, &engine, &sumw2](Int_t i, RBinIndex index) {
      if constexpr (std::is_same_v<T, RBinWithError>) {
         if (sumw2 == nullptr) {
            ret->Sumw2();
            sumw2 = ret->GetSumw2()->GetArray();
         }
         const RBinWithError &c = engine.GetBinContent(index);
         ret->GetArray()[i] = c.fSum;
         sumw2[i] = c.fSum2;
      } else {
         (void)sumw2;
         ret->GetArray()[i] = engine.GetBinContent(index);
      }
   };

   const auto &axes = engine.GetAxes();
   RBinIndexRange range;
   if (auto *regular = std::get_if<RRegularAxis>(&axes[0])) {
      const std::size_t nNormalBins = regular->GetNNormalBins();
      ret.reset(new Hist("", "", nNormalBins, regular->GetLow(), regular->GetHigh()));

      // Convert the flow bins, if enabled.
      if (regular->HasFlowBins()) {
         copyBinContent(0, RBinIndex::Underflow());
         copyBinContent(nNormalBins + 1, RBinIndex::Overflow());
      }

      // Get the range of normal bins for the loop below.
      range = regular->GetNormalRange();
   } else {
      throw std::logic_error("unimplemented axis type"); // GCOVR_EXCL_LINE
   }

   assert(ret);
   ret->SetDirectory(nullptr);

   // Convert the normal bins, accounting for TH1 numbering conventions.
   for (auto index : range) {
      copyBinContent(index.GetIndex() + 1, index);
   }

   return ret;
}

template <typename Hist>
void ConvertGlobalStatistics(Hist &h, const RHistStats &stats)
{
   h.SetEntries(stats.GetNEntries());

   Double_t hStats[4] = {
      stats.GetSumW(),
      stats.GetSumW2(),
      stats.GetDimensionStats(0).fSumWX,
      stats.GetDimensionStats(0).fSumWX2,
   };
   h.PutStats(hStats);
}
} // namespace

namespace ROOT {
namespace Experimental {
namespace Hist {

std::unique_ptr<TH1C> ConvertToTH1C(const RHistEngine<char> &engine)
{
   return ConvertToTH1Impl<TH1C>(engine);
}

std::unique_ptr<TH1S> ConvertToTH1S(const RHistEngine<short> &engine)
{
   return ConvertToTH1Impl<TH1S>(engine);
}

std::unique_ptr<TH1I> ConvertToTH1I(const RHistEngine<int> &engine)
{
   return ConvertToTH1Impl<TH1I>(engine);
}

std::unique_ptr<TH1L> ConvertToTH1L(const RHistEngine<long> &engine)
{
   return ConvertToTH1Impl<TH1L>(engine);
}

std::unique_ptr<TH1L> ConvertToTH1L(const RHistEngine<long long> &engine)
{
   return ConvertToTH1Impl<TH1L>(engine);
}

std::unique_ptr<TH1F> ConvertToTH1F(const RHistEngine<float> &engine)
{
   return ConvertToTH1Impl<TH1F>(engine);
}

std::unique_ptr<TH1D> ConvertToTH1D(const RHistEngine<double> &engine)
{
   return ConvertToTH1Impl<TH1D>(engine);
}

std::unique_ptr<TH1D> ConvertToTH1D(const RHistEngine<RBinWithError> &engine)
{
   return ConvertToTH1Impl<TH1D>(engine);
}

std::unique_ptr<TH1C> ConvertToTH1C(const RHist<char> &hist)
{
   auto ret = ConvertToTH1C(hist.GetEngine());
   ConvertGlobalStatistics(*ret, hist.GetStats());
   return ret;
}

std::unique_ptr<TH1S> ConvertToTH1S(const RHist<short> &hist)
{
   auto ret = ConvertToTH1S(hist.GetEngine());
   ConvertGlobalStatistics(*ret, hist.GetStats());
   return ret;
}

std::unique_ptr<TH1I> ConvertToTH1I(const RHist<int> &hist)
{
   auto ret = ConvertToTH1I(hist.GetEngine());
   ConvertGlobalStatistics(*ret, hist.GetStats());
   return ret;
}

std::unique_ptr<TH1L> ConvertToTH1L(const RHist<long> &hist)
{
   auto ret = ConvertToTH1L(hist.GetEngine());
   ConvertGlobalStatistics(*ret, hist.GetStats());
   return ret;
}

std::unique_ptr<TH1L> ConvertToTH1L(const RHist<long long> &hist)
{
   auto ret = ConvertToTH1L(hist.GetEngine());
   ConvertGlobalStatistics(*ret, hist.GetStats());
   return ret;
}

std::unique_ptr<TH1F> ConvertToTH1F(const RHist<float> &hist)
{
   auto ret = ConvertToTH1F(hist.GetEngine());
   ConvertGlobalStatistics(*ret, hist.GetStats());
   return ret;
}

std::unique_ptr<TH1D> ConvertToTH1D(const RHist<double> &hist)
{
   auto ret = ConvertToTH1D(hist.GetEngine());
   ConvertGlobalStatistics(*ret, hist.GetStats());
   return ret;
}

std::unique_ptr<TH1D> ConvertToTH1D(const RHist<RBinWithError> &hist)
{
   auto ret = ConvertToTH1D(hist.GetEngine());
   ConvertGlobalStatistics(*ret, hist.GetStats());
   return ret;
}

} // namespace Hist
} // namespace Experimental
} // namespace ROOT
