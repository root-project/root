#include <ROOT/RRegularAxis.hxx>

#include <benchmark/benchmark.h>

#include <random>
#include <vector>

struct RRegularAxis : public benchmark::Fixture {
   // The axis is stored and constructed in the fixture to avoid compiler optimizations in the benchmark body taking
   // advantage of the (constant) constructor parameters.
   ROOT::Experimental::RRegularAxis axis{20, 0.0, 1.0};
   std::vector<double> fNumbers;

   // Avoid GCC warning
   using benchmark::Fixture::SetUp;
   void SetUp(benchmark::State &state) final
   {
      std::mt19937 gen;
      std::uniform_real_distribution<> dis;
      fNumbers.resize(state.range(0));
      for (std::size_t i = 0; i < fNumbers.size(); i++) {
         fNumbers[i] = dis(gen);
      }
   }
};

BENCHMARK_DEFINE_F(RRegularAxis, ComputeLinearizedIndex)(benchmark::State &state)
{
   for (auto _ : state) {
      for (double number : fNumbers) {
         benchmark::DoNotOptimize(axis.ComputeLinearizedIndex(number));
      }
   }
}
BENCHMARK_REGISTER_F(RRegularAxis, ComputeLinearizedIndex)->Range(0, 32768);

BENCHMARK_MAIN();
