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
   auto copyBinContent = [&ret, &engine](Int_t i, RBinIndex index) {
      ret->GetArray()[i] = engine.GetBinContent(index);
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

std::unique_ptr<TH1I> ConvertToTH1I(const RHistEngine<int> &engine)
{
   return ConvertToTH1Impl<TH1I>(engine);
}

std::unique_ptr<TH1I> ConvertToTH1I(const RHist<int> &hist)
{
   auto ret = ConvertToTH1I(hist.GetEngine());
   ConvertGlobalStatistics(*ret, hist.GetStats());
   return ret;
}

} // namespace Hist
} // namespace Experimental
} // namespace ROOT
