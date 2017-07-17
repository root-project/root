#include "Math/Cartesian3D.h"

#include "Math/Math_vectypes.hxx"

#include "benchmark/benchmark.h"

#include <random>

static bool is_aligned(const void *__restrict__ ptr, size_t align)
{
   return (uintptr_t)ptr % align == 0;
}

static void BM_Cartesian3D_Sanity(benchmark::State &state)
{
   // Report if we forgot to turn on the explicit vectorisation in ROOT.
   if (sizeof(double) == sizeof(ROOT::Double_v) || sizeof(float) == sizeof(ROOT::Float_v))
      state.SkipWithError("Explicit vectorisation is disabled!");
}
BENCHMARK(BM_Cartesian3D_Sanity);

static void BM_Cartesian3D_CreateEmpty(benchmark::State &state)
{
   while (state.KeepRunning()) ROOT::Math::Cartesian3D<ROOT::Double_v> c;
}
BENCHMARK(BM_Cartesian3D_CreateEmpty);

template <typename T>
static void BM_Cartesian3D_Theta(benchmark::State &state)
{
   ROOT::Math::Cartesian3D<T> c(1., 2., 3.);
   // std::cout << is_aligned(&c, 16) << std::endl;
   while (state.KeepRunning()) c.Theta();
}
BENCHMARK_TEMPLATE(BM_Cartesian3D_Theta, double);
BENCHMARK_TEMPLATE(BM_Cartesian3D_Theta, ROOT::Double_v);
BENCHMARK_TEMPLATE(BM_Cartesian3D_Theta, float);
BENCHMARK_TEMPLATE(BM_Cartesian3D_Theta, ROOT::Float_v);

template <typename T>
static void BM_Cartesian3D_Phi(benchmark::State &state)
{
   ROOT::Math::Cartesian3D<T> c(1., 2., 3.);
   // std::cout << is_aligned(&c, 16) << std::endl;
   while (state.KeepRunning()) c.Phi();
}
BENCHMARK_TEMPLATE(BM_Cartesian3D_Phi, double);
BENCHMARK_TEMPLATE(BM_Cartesian3D_Phi, ROOT::Double_v);
BENCHMARK_TEMPLATE(BM_Cartesian3D_Phi, float);
BENCHMARK_TEMPLATE(BM_Cartesian3D_Phi, ROOT::Float_v);

template <typename T>
static void BM_Cartesian3D_Mag2(benchmark::State &state)
{
   ROOT::Math::Cartesian3D<T> c(1., 2., 3.);
   // std::cout << is_aligned(&c, 16) << std::endl;
   while (state.KeepRunning()) c.Mag2();
}
BENCHMARK_TEMPLATE(BM_Cartesian3D_Mag2, double);
BENCHMARK_TEMPLATE(BM_Cartesian3D_Mag2, ROOT::Double_v);

// Define our main.
BENCHMARK_MAIN();
