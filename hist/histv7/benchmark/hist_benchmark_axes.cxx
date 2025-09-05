#include <ROOT/RAxes.hxx>

#include <benchmark/benchmark.h>

#include <random>
#include <tuple>
#include <vector>

struct RAxes_Regular1 : public benchmark::Fixture {
   // The objects are stored and constructed in the fixture to avoid compiler optimizations in the benchmark body taking
   // advantage of the (constant) constructor parameters.
   ROOT::Experimental::RRegularAxis axis{20, 0.0, 1.0};
   ROOT::Experimental::Internal::RAxes axes{{axis}};
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

BENCHMARK_DEFINE_F(RAxes_Regular1, ComputeGlobalIndex)(benchmark::State &state)
{
   for (auto _ : state) {
      for (double number : fNumbers) {
         benchmark::DoNotOptimize(axes.ComputeGlobalIndex(std::make_tuple(number)));
      }
   }
}
BENCHMARK_REGISTER_F(RAxes_Regular1, ComputeGlobalIndex)->Range(0, 32768);

struct RAxes_Regular2 : public benchmark::Fixture {
   // The objects are stored and constructed in the fixture to avoid compiler optimizations in the benchmark body taking
   // advantage of the (constant) constructor parameters.
   ROOT::Experimental::RRegularAxis axis{20, 0.0, 1.0};
   ROOT::Experimental::Internal::RAxes axes{{axis, axis}};
   std::vector<double> fNumbers;

   // Avoid GCC warning
   using benchmark::Fixture::SetUp;
   void SetUp(benchmark::State &state) final
   {
      std::mt19937 gen;
      std::uniform_real_distribution<> dis;
      fNumbers.resize(2 * state.range(0));
      for (std::size_t i = 0; i < fNumbers.size(); i++) {
         fNumbers[i] = dis(gen);
      }
   }
};

BENCHMARK_DEFINE_F(RAxes_Regular2, ComputeGlobalIndex)(benchmark::State &state)
{
   for (auto _ : state) {
      for (std::size_t i = 0; i < fNumbers.size(); i += 2) {
         benchmark::DoNotOptimize(axes.ComputeGlobalIndex(std::make_tuple(fNumbers[i], fNumbers[i + 1])));
      }
   }
}
BENCHMARK_REGISTER_F(RAxes_Regular2, ComputeGlobalIndex)->Range(0, 32768);

BENCHMARK_MAIN();
