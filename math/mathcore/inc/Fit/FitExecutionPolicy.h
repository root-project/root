#include "ExecutionPolicy.hxx"
#include "ROOT/RConfig.hxx"

namespace ROOT {
namespace Fit {
namespace ExecutionPolicy {

constexpr ROOT::ExecutionPolicy
   _R__DEPRECATED_626("ROOT::Fit::ExecutionPolicy is being deprecated."
                      "Use ROOT::ExecutionPolicy::kMultiThread instead.")
      kMultithread = ROOT::ExecutionPolicy::kMultiThread;

constexpr ROOT::ExecutionPolicy
   _R__DEPRECATED_626("ROOT::Fit::ExecutionPolicy is being deprecated."
                      "Use ROOT::ExecutionPolicy::kMultiProcess instead.")
      kMultiprocess = ROOT::ExecutionPolicy::kMultiProcess;

constexpr ROOT::ExecutionPolicy _R__DEPRECATED_626(
   "ROOT::Fit::ExecutionPolicy is being deprecated."
   "Use ROOT::ExecutionPolicy::kSequential instead.")
      kSerial = ROOT::ExecutionPolicy::kSequential;

} // namespace ExecutionPolicy
} // Fit ns
} // namespace ROOT
