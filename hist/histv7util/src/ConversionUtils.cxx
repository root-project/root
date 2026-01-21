/// \file
/// \warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
/// Feedback is welcome!

#include <ROOT/Hist/ConversionUtils.hxx>
#include <ROOT/RAxisVariant.hxx>
#include <ROOT/RRegularAxis.hxx>

#include <TAxis.h>

namespace ROOT {
namespace Experimental {
namespace Hist {
namespace Internal {

void ConvertAxis(TAxis &dst, const RAxisVariant &src)
{
   if (auto *regular = src.GetRegularAxis()) {
      dst.Set(regular->GetNNormalBins(), regular->GetLow(), regular->GetHigh());
   } else {
      throw std::logic_error("unimplemented axis type in ConvertAxis"); // GCOVR_EXCL_LINE
   }
}

} // namespace Internal
} // namespace Hist
} // namespace Experimental
} // namespace ROOT
