#ifndef TMVA_RINFERENCEUTILS
#define TMVA_RINFERENCEUTILS

#include <utility> // std::forward, std::index_sequence

namespace TMVA {
namespace Experimental {

namespace Internal {

/// Compute helper
template <typename I, typename T, typename F>
class ComputeHelper;

template <std::size_t... N, typename T, typename F>
class ComputeHelper<std::index_sequence<N...>, T, F> {
   template <std::size_t Idx>
   using AlwaysT = T;
   F fFunc;

public:
   ComputeHelper(F &&f) : fFunc(std::forward<F>(f)) {}
   auto operator()(AlwaysT<N>... args) -> decltype(fFunc.Compute({args...})) { return fFunc.Compute({args...}); }
};

} // namespace Internal

/// Helper to pass TMVA model to RDataFrame.Define nodes
template <std::size_t N, typename T, typename F>
auto Compute(F &&f) -> Internal::ComputeHelper<std::make_index_sequence<N>, T, F>
{
   return Internal::ComputeHelper<std::make_index_sequence<N>, T, F>(std::forward<F>(f));
}

} // namespace Experimental
} // namespace TMVA

#endif // TMVA_RINFERENCEUTILS
