#include "Math/Cartesian3D.h"

#include "Math/Math_vectypes.hxx"

#include "benchmark/benchmark.h"

#include <random>

static bool is_aligned(const void *__restrict__ ptr, size_t align)
{
   return (uintptr_t)ptr % align == 0;
}

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
BENCHMARK_TEMPLATE(BM_Cartesian3D_Theta, double)->Range(8, 8 << 10)->Complexity(benchmark::o1);
BENCHMARK_TEMPLATE(BM_Cartesian3D_Theta, ROOT::Double_v)->Range(8, 8 << 10)->Complexity(benchmark::o1);
BENCHMARK_TEMPLATE(BM_Cartesian3D_Theta, float)->Range(8, 8 << 10)->Complexity(benchmark::o1);
BENCHMARK_TEMPLATE(BM_Cartesian3D_Theta, ROOT::Float_v)->Range(8, 8 << 10)->Complexity(benchmark::o1);

template <typename T>
static void BM_Cartesian3D_Phi(benchmark::State &state)
{
   ROOT::Math::Cartesian3D<T> c(1., 2., 3.);
   // std::cout << is_aligned(&c, 16) << std::endl;
   while (state.KeepRunning()) c.Phi();
   state.SetComplexityN(state.range(0));
}
BENCHMARK_TEMPLATE(BM_Cartesian3D_Phi, double)->Range(8, 8 << 10)->Complexity(benchmark::o1);
BENCHMARK_TEMPLATE(BM_Cartesian3D_Phi, ROOT::Double_v)->Range(8, 8 << 10)->Complexity(benchmark::o1);
BENCHMARK_TEMPLATE(BM_Cartesian3D_Phi, float)->Range(8, 8 << 10)->Complexity(benchmark::o1);
BENCHMARK_TEMPLATE(BM_Cartesian3D_Phi, ROOT::Float_v)->Range(8, 8 << 10)->Complexity(benchmark::o1);

template <typename T>
static void BM_Cartesian3D_Mag2(benchmark::State &state)
{
   ROOT::Math::Cartesian3D<T> c(1., 2., 3.);
   // std::cout << is_aligned(&c, 16) << std::endl;
   while (state.KeepRunning()) c.Mag2();
   state.SetComplexityN(state.range(0));
}
BENCHMARK_TEMPLATE(BM_Cartesian3D_Mag2, double)->Range(8, 8 << 10)->Complexity(benchmark::o1);
BENCHMARK_TEMPLATE(BM_Cartesian3D_Mag2, ROOT::Double_v)->Range(8, 8 << 10)->Complexity(benchmark::o1);

// Define our main.
BENCHMARK_MAIN();
