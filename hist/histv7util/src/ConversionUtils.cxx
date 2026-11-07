/// \file
/// \warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
/// Feedback is welcome!

#include <ROOT/Hist/ConversionUtils.hxx>
#include <ROOT/RAxisVariant.hxx>
#include <ROOT/RCategoricalAxis.hxx>
#include <ROOT/RRegularAxis.hxx>
#include <ROOT/RVariableBinAxis.hxx>

#include <TAxis.h>

namespace ROOT {
namespace Experimental {
namespace Hist {
namespace Internal {

void ConvertAxis(TAxis &dst, const RAxisVariant &src)
{
   if (auto *regular = src.GetRegularAxis()) {
      dst.Set(regular->GetNNormalBins(), regular->GetLow(), regular->GetHigh());
   } else if (auto *variable = src.GetVariableBinAxis()) {
      dst.Set(variable->GetNNormalBins(), variable->GetBinEdges().data());
   } else if (auto *categorical = src.GetCategoricalAxis()) {
      const auto &categories = categorical->GetCategories();
      dst.Set(categories.size(), 0, categories.size());

      for (std::size_t i = 0; i < categories.size(); i++) {
         dst.SetBinLabel(i + 1, categories[i].c_str());
      }
   } else {
      throw std::logic_error("unimplemented axis type in ConvertAxis"); // GCOVR_EXCL_LINE
   }
}

} // namespace Internal
} // namespace Hist
} // namespace Experimental
} // namespace ROOT
