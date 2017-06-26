#include "Math/Cartesian3D.h"

#include "Math/Math_vectypes.hxx"

#include "benchmark/benchmark.h"

static void BM_Cartesian3D_CreateEmpty(benchmark::State &state)
{
   while (state.KeepRunning()) ROOT::Math::Cartesian3D<double> c;
}
BENCHMARK(BM_Cartesian3D_CreateEmpty);

template <typename T>
static void BM_Cartesian3D_Theta(benchmark::State &state)
{
   ROOT::Math::Cartesian3D<T> c(1, 2, 3);
   while (state.KeepRunning()) c.Theta();
}
BENCHMARK_TEMPLATE(BM_Cartesian3D_Theta, double)->Range(8, 8 << 10);
BENCHMARK_TEMPLATE(BM_Cartesian3D_Theta, ROOT::Double_v)->Range(8, 8 << 10);

// Define our main.
BENCHMARK_MAIN();
