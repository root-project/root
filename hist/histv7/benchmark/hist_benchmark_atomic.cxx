#include <ROOT/RBinWithError.hxx>
#include <ROOT/RHistUtils.hxx>

// Note: In principle, the benchmark library supports multithreaded benchmarks. However, the output has historically
// been confusing (wrong): https://github.com/google/benchmark/issues/1834 This was changed / fixed in version 1.9.0:
// https://github.com/google/benchmark/pull/1836 Even then, the minimum benchmark time is not correctly respected (in
// the way I would expect): https://github.com/google/benchmark/issues/2117
// Moreover, measuring contention heavily depends on the machine, cache coherency, tread placement / binding, and other
// factors. Reproducibility is very hard to achieve with the limited control that the benchmark library offers.
// For these reasons, the following benchmarks are not automatically run with multiple threads to benchmark contention.
// Care should be taken when manually changing and trying to interpret the results!
// To keep this possiblity though, the benchmarks MUST NOT use benchmark::DoNotOptimize() on non-const references to
// shared variables: Older compilers (GCC before version 12.1.0) will load and store the value through the reference,
// which leads to data races!
#include <benchmark/benchmark.h>

#include <cstddef>

struct RHistAtomic_int : public benchmark::Fixture {
   int fAtomic = 0;
};

BENCHMARK_DEFINE_F(RHistAtomic_int, Add)(benchmark::State &state)
{
   for (auto _ : state) {
      fAtomic += 1;
      benchmark::ClobberMemory();
   }
}
BENCHMARK_REGISTER_F(RHistAtomic_int, Add);

BENCHMARK_DEFINE_F(RHistAtomic_int, AtomicAdd)(benchmark::State &state)
{
   for (auto _ : state) {
      ROOT::Experimental::Internal::AtomicAdd(&fAtomic, 1);
      benchmark::ClobberMemory();
   }
}
BENCHMARK_REGISTER_F(RHistAtomic_int, AtomicAdd);

struct RHistAtomic_float : public benchmark::Fixture {
   float fAtomic = 0;
};

BENCHMARK_DEFINE_F(RHistAtomic_float, Add)(benchmark::State &state)
{
   for (auto _ : state) {
      fAtomic += 1.0f;
      benchmark::ClobberMemory();
   }
}
BENCHMARK_REGISTER_F(RHistAtomic_float, Add);

BENCHMARK_DEFINE_F(RHistAtomic_float, AtomicAdd)(benchmark::State &state)
{
   for (auto _ : state) {
      ROOT::Experimental::Internal::AtomicAdd(&fAtomic, 1.0f);
      benchmark::ClobberMemory();
   }
}
BENCHMARK_REGISTER_F(RHistAtomic_float, AtomicAdd);

struct RHistAtomic_double : public benchmark::Fixture {
   double fAtomic = 0;
};

BENCHMARK_DEFINE_F(RHistAtomic_double, Add)(benchmark::State &state)
{
   for (auto _ : state) {
      fAtomic += 1.0;
      benchmark::ClobberMemory();
   }
}
BENCHMARK_REGISTER_F(RHistAtomic_double, Add);

BENCHMARK_DEFINE_F(RHistAtomic_double, AtomicAdd)(benchmark::State &state)
{
   for (auto _ : state) {
      ROOT::Experimental::Internal::AtomicAdd(&fAtomic, 1.0);
      benchmark::ClobberMemory();
   }
}
BENCHMARK_REGISTER_F(RHistAtomic_double, AtomicAdd);

struct RBinWithError : public benchmark::Fixture {
   ROOT::Experimental::RBinWithError fBin;
};

BENCHMARK_DEFINE_F(RBinWithError, Inc)(benchmark::State &state)
{
   for (auto _ : state) {
      fBin++;
      benchmark::ClobberMemory();
   }
}
BENCHMARK_REGISTER_F(RBinWithError, Inc);

BENCHMARK_DEFINE_F(RBinWithError, AtomicInc)(benchmark::State &state)
{
   for (auto _ : state) {
      fBin.AtomicInc();
      benchmark::ClobberMemory();
   }
}
BENCHMARK_REGISTER_F(RBinWithError, AtomicInc);

BENCHMARK_DEFINE_F(RBinWithError, Add)(benchmark::State &state)
{
   for (auto _ : state) {
      fBin += 1.0;
      benchmark::ClobberMemory();
   }
}
BENCHMARK_REGISTER_F(RBinWithError, Add);

BENCHMARK_DEFINE_F(RBinWithError, AtomicAdd)(benchmark::State &state)
{
   for (auto _ : state) {
      fBin.AtomicAdd(1.0);
      benchmark::ClobberMemory();
   }
}
BENCHMARK_REGISTER_F(RBinWithError, AtomicAdd);

BENCHMARK_MAIN();
