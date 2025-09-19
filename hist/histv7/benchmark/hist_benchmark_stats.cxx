#include <ROOT/RHistStats.hxx>
#include <ROOT/RWeight.hxx>

#include <benchmark/benchmark.h>

#include <random>
#include <vector>

struct RHistStats1 : public benchmark::Fixture {
   // The object is stored and constructed in the fixture to avoid compiler optimizations in the benchmark body taking
   // advantage of the (constant) constructor parameters.
   ROOT::Experimental::RHistStats stats{1};
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

BENCHMARK_DEFINE_F(RHistStats1, Fill)(benchmark::State &state)
{
   for (auto _ : state) {
      for (double number : fNumbers) {
         stats.Fill(number);
      }
   }
}
BENCHMARK_REGISTER_F(RHistStats1, Fill)->Range(0, 32768);

BENCHMARK_DEFINE_F(RHistStats1, FillWeight)(benchmark::State &state)
{
   for (auto _ : state) {
      for (double number : fNumbers) {
         stats.Fill(number, ROOT::Experimental::RWeight(0.8));
      }
   }
}
BENCHMARK_REGISTER_F(RHistStats1, FillWeight)->Range(0, 32768);

struct RHistStats2 : public benchmark::Fixture {
   // The object is stored and constructed in the fixture to avoid compiler optimizations in the benchmark body taking
   // advantage of the (constant) constructor parameters.
   ROOT::Experimental::RHistStats stats{2};
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

BENCHMARK_DEFINE_F(RHistStats2, Fill)(benchmark::State &state)
{
   for (auto _ : state) {
      for (std::size_t i = 0; i < fNumbers.size(); i += 2) {
         stats.Fill(fNumbers[i], fNumbers[i + 1]);
      }
   }
}
BENCHMARK_REGISTER_F(RHistStats2, Fill)->Range(0, 32768);

BENCHMARK_DEFINE_F(RHistStats2, FillWeight)(benchmark::State &state)
{
   for (auto _ : state) {
      for (std::size_t i = 0; i < fNumbers.size(); i += 2) {
         stats.Fill(fNumbers[i], fNumbers[i + 1], ROOT::Experimental::RWeight(0.8));
      }
   }
}
BENCHMARK_REGISTER_F(RHistStats2, FillWeight)->Range(0, 32768);

BENCHMARK_MAIN();
