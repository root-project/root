#include "Math/Cartesian3D.h"

#include "Math/Types.h"

#include "RandomNumberEngine.h"

#include "benchmark/benchmark.h"

#include <sstream>

// FIXME: Reduce the boilerplate code of each benchmark. Maybe setting up a Benchmark Fixture will help.

static void BM_Cartesian3D_Sanity(benchmark::State &state)
{
   // Report if we forgot to turn on the explicit vectorisation in ROOT.
   if (sizeof(double) == sizeof(ROOT::Double_v) || sizeof(float) == sizeof(ROOT::Float_v))
      state.SkipWithError("Explicit vectorisation is disabled!");

#ifndef NDEBUG
   // If we compile gbenchmark in debug mode it asserts because this particular benchmark doesn't have
   // state.KeepRunning() clause. Here this is meant to perform a simple sanity check for the expected configuration.
   while (state.KeepRunning()) {
   }
#endif
}
BENCHMARK(BM_Cartesian3D_Sanity);

static void BM_Cartesian3D_CreateEmpty(benchmark::State &state)
{
   while (state.KeepRunning()) ROOT::Math::Cartesian3D<ROOT::Double_v> c;
}
BENCHMARK(BM_Cartesian3D_CreateEmpty);

static void ComputeProcessedEntities(benchmark::State &state, size_t Size, double checksum)
{
   const size_t items_processed = state.iterations() * state.range(0);
   state.SetItemsProcessed(items_processed);
   state.SetBytesProcessed(items_processed * Size);
   std::stringstream ss;
   ss << checksum;
   state.SetLabel(ss.str());
}

template <typename T>
static void BM_Cartesian3D_Theta(benchmark::State &state)
{
   using namespace ROOT::Benchmark;
   // We want the same random numbers for a benchmark family.
   SetSeed(1);
   // Allocate N points
   constexpr size_t VecSize = TypeSize<T>::Get();
   typename Data<T>::Vector Points(state.range(0) / VecSize);
   ROOT::Math::Cartesian3D<T> c;
   T checksum = T();
   while (state.KeepRunning()) {
      checksum = T();
      for (auto &p : Points) {
         c.SetCoordinates(p.X, p.Y, p.Z);
         checksum += c.Theta();
      }
   }
   ComputeProcessedEntities(state, sizeof(T), VecTraits<T>::Sum(checksum));
}
BENCHMARK_TEMPLATE(BM_Cartesian3D_Theta, double)->Range(8 << 10, 8 << 20);
BENCHMARK_TEMPLATE(BM_Cartesian3D_Theta, ROOT::Double_v)->Range(8 << 10, 8 << 20);
// BENCHMARK_TEMPLATE(BM_Cartesian3D_Theta, float)->Range(8 << 10, 8 << 20);
// BENCHMARK_TEMPLATE(BM_Cartesian3D_Theta, ROOT::Float_v)->Range(8 << 10, 8 << 20);

template <typename T>
static void BM_Cartesian3D_Phi(benchmark::State &state)
{
   using namespace ROOT::Benchmark;
   // We want the same random numbers for a benchmark family.
   SetSeed(2);
   // Allocate N points
   constexpr size_t VecSize = TypeSize<T>::Get();
   typename Data<T>::Vector Points(state.range(0) / VecSize);
   ROOT::Math::Cartesian3D<T> c;
   T checksum = T();
   while (state.KeepRunning()) {
      checksum = T();
      for (auto &p : Points) {
         c.SetCoordinates(p.X, p.Y, p.Z);
         checksum += c.Phi();
      }
   }
   ComputeProcessedEntities(state, sizeof(T), VecTraits<T>::Sum(checksum));
}
BENCHMARK_TEMPLATE(BM_Cartesian3D_Phi, double)->Range(8 << 10, 8 << 20);
BENCHMARK_TEMPLATE(BM_Cartesian3D_Phi, ROOT::Double_v)->Range(8 << 10, 8 << 20);
// BENCHMARK_TEMPLATE(BM_Cartesian3D_Phi, float)->Range(8 << 10, 8 << 20);
// BENCHMARK_TEMPLATE(BM_Cartesian3D_Phi, ROOT::Float_v)->Range(8 << 10, 8 << 20);

template <typename T>
static void BM_Cartesian3D_Mag2(benchmark::State &state)
{
   using namespace ROOT::Benchmark;
   // We want the same random numbers for a benchmark family.
   SetSeed(3);
   // Allocate N points
   constexpr size_t VecSize = TypeSize<T>::Get();
   typename Data<T>::Vector Points(state.range(0) / VecSize);
   ROOT::Math::Cartesian3D<T> c;
   T checksum = T();
   while (state.KeepRunning()) {
      checksum = T();
      for (auto &p : Points) {
         c.SetCoordinates(p.X, p.Y, p.Z);
         checksum += c.Mag2();
      }
   }

   ComputeProcessedEntities(state, sizeof(T), VecTraits<T>::Sum(checksum));
}
BENCHMARK_TEMPLATE(BM_Cartesian3D_Mag2, double)->Range(8 << 10, 8 << 20);
BENCHMARK_TEMPLATE(BM_Cartesian3D_Mag2, ROOT::Double_v)->Range(8 << 10, 8 << 20);
// BENCHMARK_TEMPLATE(BM_Cartesian3D_Mag2, float)->Range(8 << 10, 8 << 20);
// BENCHMARK_TEMPLATE(BM_Cartesian3D_Mag2, ROOT::Float_v)->Range(8 << 10, 8 << 20);

template <typename T>
static void BM_Cartesian3D_Scale(benchmark::State &state)
{
   using namespace ROOT::Benchmark;
   // We want the same random numbers for a benchmark family.
   SetSeed(4);
   // Allocate N points
   constexpr size_t VecSize = TypeSize<T>::Get();
   typename Data<T>::Vector Points(state.range(0) / VecSize);
   ROOT::Math::Cartesian3D<T> c;
   T checksum = T();
   while (state.KeepRunning()) {
      checksum = T();
      for (auto &p : Points) {
         c.SetCoordinates(p.X, p.Y, p.Z);
         c.Scale(p.X);
         checksum += c.x() + c.y() + c.z();
      }
   }

   ComputeProcessedEntities(state, sizeof(T), VecTraits<T>::Sum(checksum));
}
BENCHMARK_TEMPLATE(BM_Cartesian3D_Scale, double)->Range(8 << 10, 8 << 20);
BENCHMARK_TEMPLATE(BM_Cartesian3D_Scale, ROOT::Double_v)->Range(8 << 10, 8 << 20);

// Define our main.
BENCHMARK_MAIN();
