// Cross-TU performance check for VTableOverlay.
//
// The dispatch loop (OverlayDispatchOnce) lives in TestSharedLib, so the
// call site cannot devirtualize or inline against the derived type known
// here. That makes the measurement honest: it reflects a real AOT caller
// in another library dispatching through the (possibly overlaid) vtable.
//
// VTableOverlay does NOT make the call faster -- it is not devirtualization
// or inlining; it only swaps the function pointer installed at a vtable
// slot. The claim under test is that the swap is *free at the call site*:
// the overlaid path runs the same vtable indirection as a plain virtual
// call, so it costs the same. That zero-overhead property is what enables
// language bindings to redirect dispatch into Python (or another runtime)
// without paying any per-call cost over the C++ virtual baseline.

#include "CppInterOp/CppInterOp.h"
#include "CppInterOp/CppInterOpTypes.h"
#include "TestSharedLib/TestSharedLib.h"

#include "Utils.h"
#include "PerfCompare.h"
#include "gtest/gtest.h"

#include <benchmark/benchmark.h>

#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

struct Impl : OverlayBase {
  [[gnu::noinline]] int frob(int x) override { return x + 1; }
};
extern "C" int xtu_replacement(void* /*self*/, int x) { return x + 100; }

// Reflect OverlayBase (definition mirrors TestSharedLib) so the overlay is
// driven through the public reflected API -- the language-binding setting.
Cpp::DeclRef ReflectOverlayBase() {
  Cpp::CreateInterpreter({});
  Cpp::Declare("struct OverlayBase {"
               "  OverlayBase();"
               "  virtual ~OverlayBase();"
               "  virtual int frob(int x);"
               "};");
  return Cpp::GetNamed("OverlayBase");
}

Cpp::FuncRef Frob(Cpp::DeclRef scope) {
  std::vector<Cpp::FuncRef> methods;
  Cpp::GetClassMethods(scope, methods);
  for (auto m : methods)
    if (Cpp::GetName(Cpp::DeclRef{m.data}) == "frob")
      return m;
  return nullptr;
}

Cpp::UniqueVTableOverlay OverlayFrob(void* inst, Cpp::DeclRef base) {
  return Cpp::MakeUniqueVTableOverlay(
      inst, base, {{Frob(base), TestUtils::BitCastFn<void*>(&xtu_replacement)}});
}

} // namespace

TEST(VTableOverlayCrossTU, OverlayDispatchesAcrossDSO) {
  Impl impl;
  auto base = ReflectOverlayBase();
  ASSERT_NE(base, nullptr);
  auto ov = OverlayFrob(&impl, base);
  ASSERT_TRUE(ov);
  // The call site is in TestSharedLib; the overlay still takes effect.
  EXPECT_EQ(OverlayDispatchOnce(&impl, 7), 107);
}

static void BM_XTU_BareVirtual(benchmark::State& state) {
  Impl impl;
  for (auto _ : state)
    benchmark::DoNotOptimize(OverlayDispatchOnce(&impl, 7));
}
BENCHMARK(BM_XTU_BareVirtual);

static void BM_XTU_OverlayDirectFn(benchmark::State& state) {
  Impl impl;
  auto base = ReflectOverlayBase();
  auto ov = OverlayFrob(&impl, base);
  for (auto _ : state)
    benchmark::DoNotOptimize(OverlayDispatchOnce(&impl, 7));
}
BENCHMARK(BM_XTU_OverlayDirectFn);

TEST(VTableOverlayCrossTU, OverlayAddsNoPerCallCost) {
#if !defined(NDEBUG) || defined(__SANITIZE_ADDRESS__)
  GTEST_SKIP() << "Perf assertions need a Release, non-sanitizer build.";
#endif
#ifdef __APPLE__
  GTEST_SKIP() << "Flaky on macOS runners (ratio noise around the 0.9 floor).";
#endif
  EXPECT_NOT_SLOWER_THAN(BM_XTU_OverlayDirectFn, BM_XTU_BareVirtual);
}

// Pin the perf claim that motivates n_extra_prefix_slots: a thunk
// reading per-instance state via the extra slot is measurably faster
// than the alternative bindings would otherwise reach for (a
// process-global pointer registry keyed by `self`). The dispatched
// work is identical in both thunks -- only the state-lookup path
// differs -- so the gap measured here is the lookup itself.
namespace {
struct Handler {
  int v;
  [[gnu::noinline]] int add(int x) { return v + x; }
};

extern "C" int xtu_thunk_extra_slot(void* self, int x) {
  auto* h = static_cast<Handler*>(Cpp::VTableOverlayExtraSlot(self, 0));
  return h->add(x);
}

// Bench-only stand-in for the naive alternative. Production code does
// not pay a global mutex per dispatch -- that is the contrast the
// EXPECT_AT_LEAST_N_TIMES_FASTER assertion below pins down.
std::unordered_map<void*, Handler*> NaiveHandlerMap;
std::mutex NaiveHandlerMapMutex;

extern "C" int xtu_thunk_global_map(void* self, int x) {
  Handler* h;
  {
    std::lock_guard<std::mutex> lk(NaiveHandlerMapMutex);
    h = NaiveHandlerMap[self];
  }
  return h->add(x);
}

Handler g_handler{100};
} // namespace

TEST(VTableOverlayCrossTU, ExtraPrefixSlotDispatch) {
  Impl impl;
  auto base = ReflectOverlayBase();
  ASSERT_NE(base, nullptr);
  auto ov = Cpp::MakeUniqueVTableOverlay(
      &impl, base,
      {{Frob(base), TestUtils::BitCastFn<void*>(&xtu_thunk_extra_slot)}},
      /*n_extra_prefix_slots=*/1);
  ASSERT_TRUE(ov);
  Cpp::VTableOverlayExtraSlot(&impl, 0) = &g_handler;
  EXPECT_EQ(OverlayDispatchOnce(&impl, 7), g_handler.v + 7);
}

static void BM_XTU_OverlayExtraSlot(benchmark::State& state) {
  Impl impl;
  auto base = ReflectOverlayBase();
  auto ov = Cpp::MakeUniqueVTableOverlay(
      &impl, base,
      {{Frob(base), TestUtils::BitCastFn<void*>(&xtu_thunk_extra_slot)}},
      /*n_extra_prefix_slots=*/1);
  Cpp::VTableOverlayExtraSlot(&impl, 0) = &g_handler;
  for (auto _ : state)
    benchmark::DoNotOptimize(OverlayDispatchOnce(&impl, 7));
}
BENCHMARK(BM_XTU_OverlayExtraSlot);

static void BM_XTU_OverlayGlobalMap(benchmark::State& state) {
  Impl impl;
  auto base = ReflectOverlayBase();
  auto ov = Cpp::MakeUniqueVTableOverlay(
      &impl, base,
      {{Frob(base), TestUtils::BitCastFn<void*>(&xtu_thunk_global_map)}});
  {
    std::lock_guard<std::mutex> lk(NaiveHandlerMapMutex);
    NaiveHandlerMap[&impl] = &g_handler;
  }
  for (auto _ : state)
    benchmark::DoNotOptimize(OverlayDispatchOnce(&impl, 7));
  {
    std::lock_guard<std::mutex> lk(NaiveHandlerMapMutex);
    NaiveHandlerMap.erase(&impl);
  }
}
BENCHMARK(BM_XTU_OverlayGlobalMap);

TEST(VTableOverlayCrossTU, ExtraSlotBeatsGlobalMap) {
#if !defined(NDEBUG) || defined(__SANITIZE_ADDRESS__)
  GTEST_SKIP() << "Perf assertions need a Release, non-sanitizer build.";
#endif
  // The map path is hash + mutex per dispatch; the slot path is one mov.
  // Even with a fast hash the gap is real; require >= 2x to ride out jitter.
  EXPECT_AT_LEAST_N_TIMES_FASTER(BM_XTU_OverlayExtraSlot,
                                 BM_XTU_OverlayGlobalMap, 2.0);
}

// Confirms on_destroy is destruction-only: dispatch through any other
// slot pays nothing. Reported for trend visibility, with no
// EXPECT_*_FASTER assertion: the dtor wrapper patches the deleting-dtor
// slot, not the dispatched method's slot, so the per-call cost is
// structurally unchanged (see MakeVTableOverlay). The ~30% run-to-run
// jitter on sub-10 ns measurements on CI exceeds any tolerance that
// would still catch a real regression, so a relative assertion here is
// noise, not signal.
extern "C" void xtu_noop_cleanup(void* /*inst*/, void* /*data*/) {}

static void BM_XTU_OverlayWithDtorHook(benchmark::State& state) {
  Impl impl;
  auto base = ReflectOverlayBase();
  void* fn = TestUtils::BitCastFn<void*>(&xtu_replacement);
  Cpp::ConstFuncRef method = Frob(base);
  auto* ov = Cpp::MakeVTableOverlay(
      &impl, base, &method, &fn, /*n=*/1,
      /*n_extra_prefix_slots=*/0,
      /*on_destroy=*/&xtu_noop_cleanup,
      /*cleanup_data=*/nullptr);
  for (auto _ : state)
    benchmark::DoNotOptimize(OverlayDispatchOnce(&impl, 7));
  Cpp::DestroyVTableOverlay(ov);
}
BENCHMARK(BM_XTU_OverlayWithDtorHook);

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  benchmark::Initialize(&argc, argv);
  // Print formatted benchmark output whenever the perf-assertion tests
  // themselves would run -- Debug / sanitizer numbers are noise.
#if defined(NDEBUG) && !defined(__SANITIZE_ADDRESS__)
  benchmark::RunSpecifiedBenchmarks();
#endif
  return RUN_ALL_TESTS();
}
