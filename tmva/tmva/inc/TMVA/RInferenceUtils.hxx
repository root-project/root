#ifndef TMVA_RINFERENCEUTILS
#define TMVA_RINFERENCEUTILS

#include <utility> // std::forward, std::index_sequence
#include <vector>

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
   // The inputs are explicitly wrapped in a std::vector: with the batch-inference
   // std::span overloads of Compute() around, a braced-init-list argument would be
   // ambiguous between the single-event vector overload and the batch span overload.
   auto operator()(AlwaysT<N>... args) -> decltype(fFunc.Compute(std::vector<T>{args...}))
   {
      return fFunc.Compute(std::vector<T>{args...});
   }
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
