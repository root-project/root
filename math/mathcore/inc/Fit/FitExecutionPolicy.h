#include "ExecutionPolicy.hxx"
#include "ROOT/RConfig.hxx"

namespace ROOT {
namespace Fit {
namespace ExecutionPolicy {

constexpr ROOT::Internal::ExecutionPolicy
   _R__DEPRECATED_626("ROOT::Fit::ExecutionPolicy is being deprecated."
                      "Use ROOT::Internal::ExecutionPolicy::kMultiThread instead.")
      kMultithread = ROOT::Internal::ExecutionPolicy::kMultiThread;

constexpr ROOT::Internal::ExecutionPolicy
   _R__DEPRECATED_626("ROOT::Fit::ExecutionPolicy is being deprecated."
                      "Use ROOT::Internal::ExecutionPolicy::kMultiProcess instead.")
      kMultiprocess = ROOT::Internal::ExecutionPolicy::kMultiProcess;

constexpr ROOT::Internal::ExecutionPolicy _R__DEPRECATED_626(
   "ROOT::Fit::ExecutionPolicy is being deprecated."
   "Use ROOT::Internal::ExecutionPolicy::kSequential instead.")
      kSerial = ROOT::Internal::ExecutionPolicy::kSequential;

} // namespace ExecutionPolicy
} // Fit ns
} // namespace ROOT
