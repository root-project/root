#include "Math/Cartesian3D.h"

#include "Math/Math_vectypes.hxx"

#include "RandomNumberEngine.h"

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

// template <typename T> class Cartesian3D { T x; T y; T z; /*...*/ };
// // ROOT::Double_v -> double[2]
// template <> class Cartesian3D<double[2]> { double[2] x; double[2] y; double[2] z; /*...*/ };

// template <> class Cartesian3D<double[2]> { double[2] x = 1.; double[2] y = 2.; double[2] z = 3.; /*...*/ };

static void ComputeProcessedEntities(benchmark::State &state, size_t Size)
{
   const size_t items_processed = state.iterations() * state.range(0);
   state.SetItemsProcessed(items_processed);
   state.SetBytesProcessed(items_processed * Size);
}

template <typename T>
static void BM_Cartesian3D_Theta(benchmark::State &state)
{
   using namespace ROOT::Benchmark;
   // Allocate N points
   typename Data<T>::Vector Points(state.range(0));
   constexpr size_t VecSize = TypeSize<T>::Get();
   ROOT::Math::Cartesian3D<T> c;
   while (state.KeepRunning())
      for (size_t i = 0, e = Points.size(); i < e; i += VecSize) {
         c.SetCoordinates(Points[i].X, Points[i].Y, Points[i].Z);
         c.Theta();
      }
   ComputeProcessedEntities(state, sizeof(T));
}
BENCHMARK_TEMPLATE(BM_Cartesian3D_Theta, double)->Range(1 << 0, 1 << 10);
BENCHMARK_TEMPLATE(BM_Cartesian3D_Theta, ROOT::Double_v)->Range(1 << 0, 1 << 10);
BENCHMARK_TEMPLATE(BM_Cartesian3D_Theta, float)->Range(1 << 0, 1 << 10);
BENCHMARK_TEMPLATE(BM_Cartesian3D_Theta, ROOT::Float_v)->Range(1 << 0, 1 << 10);

template <typename T>
static void BM_Cartesian3D_Phi(benchmark::State &state)
{
   using namespace ROOT::Benchmark;
   // Allocate N points
   typename Data<T>::Vector Points(state.range(0));
   constexpr size_t VecSize = TypeSize<T>::Get();
   ROOT::Math::Cartesian3D<T> c;
   while (state.KeepRunning())
      for (size_t i = 0, e = Points.size(); i < e; i += VecSize) {
         c.SetCoordinates(Points[i].X, Points[i].Y, Points[i].Z);
         c.Phi();
      }
   ComputeProcessedEntities(state, sizeof(T));
}
BENCHMARK_TEMPLATE(BM_Cartesian3D_Phi, double)->Range(1 << 0, 1 << 10);
BENCHMARK_TEMPLATE(BM_Cartesian3D_Phi, ROOT::Double_v)->Range(1 << 0, 1 << 10);
BENCHMARK_TEMPLATE(BM_Cartesian3D_Phi, float)->Range(1 << 0, 1 << 10);
BENCHMARK_TEMPLATE(BM_Cartesian3D_Phi, ROOT::Float_v)->Range(1 << 0, 1 << 10);

template <typename T>
static void BM_Cartesian3D_Mag2(benchmark::State &state)
{
   using namespace ROOT::Benchmark;
   // Allocate N points
   typename Data<T>::Vector Points(state.range(0));
   constexpr size_t VecSize = TypeSize<T>::Get();
   ROOT::Math::Cartesian3D<T> c;
   while (state.KeepRunning())
      for (size_t i = 0, e = Points.size(); i < e; i += VecSize) {
         c.SetCoordinates(Points[i].X, Points[i].Y, Points[i].Z);
         c.Mag2();
      }
   ComputeProcessedEntities(state, sizeof(T));
}
BENCHMARK_TEMPLATE(BM_Cartesian3D_Mag2, double)->Range(1 << 0, 1 << 10);
BENCHMARK_TEMPLATE(BM_Cartesian3D_Mag2, ROOT::Double_v)->Range(1 << 0, 1 << 10);

// Define our main.
BENCHMARK_MAIN();
