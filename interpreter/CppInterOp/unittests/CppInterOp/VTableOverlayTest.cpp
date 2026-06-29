#include "CppInterOp/CppInterOp.h"
#include "CppInterOp/CppInterOpTypes.h"

#include "Utils.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace {

// Parallel test-TU definitions of A / B used by the OverlayThroughHierarchy
// test. Layout and ABI come from the host compiler here; the same struct
// shape is declared at interpreter level inside the test so reflection
// computes matching vtable slot indices. Virtual destructors keep the
// vtable layout in line with the other tests here (Itanium D1/D0 pair
// before user virtuals; MSVC single deleting-dtor).
struct A {
  int m_x;
  A(int x) : m_x(x) {}
  virtual ~A() {}
  virtual int method() { return m_x; }
};

struct B : A {
  double m_d;
  B(int x, double d) : A(x), m_d(d) {}
  int method() override { return static_cast<int>(m_d); }
};

// Replacement functions installed into vtable slots. Non-static on purpose
// (vtable slots take __thiscall on 32-bit MSVC); the casts alias the real
// object's layout through an unrelated `this` -- the point of the test.
// NOLINTBEGIN(readability-convert-member-functions-to-static, cppcoreguidelines-pro-type-reinterpret-cast)
struct Repl {
  int negate(int x) { return -x; }
  int twice(int x) { return x * 2; }
  // Reads the object via `this`: layout is { vptr, int value } on every
  // ABI we target, so the data member sits at sizeof(void*).
  int read_value(int x) {
    return *reinterpret_cast<int*>(reinterpret_cast<char*>(this) +
                                   sizeof(void*)) +
           x;
  }
  int foo() { return reinterpret_cast<A*>(this)->m_x + 10; }
  int bar() {
    B* b = reinterpret_cast<B*>(this);
    return static_cast<int>(b->m_x + b->m_d);
  }
};
// NOLINTEND(readability-convert-member-functions-to-static, cppcoreguidelines-pro-type-reinterpret-cast)

template <class MFP> void* MethodAddr(MFP mfp) {
  static_assert(sizeof(MFP) >= sizeof(void*), "unexpected MFP layout");
  void* addr;
  std::memcpy(static_cast<void*>(&addr), &mfp, sizeof(addr));
  return addr;
}

#if defined(_MSC_VER) && defined(_M_IX86)
#  define CPPINTEROP_VTABLE_SLOT_CC __thiscall
#else
#  define CPPINTEROP_VTABLE_SLOT_CC
#endif

int call_slot_no_arg(void* inst, int slot) {
  void** vptr = *reinterpret_cast<void***>(inst);
  return TestUtils::BitCastFn<int (CPPINTEROP_VTABLE_SLOT_CC *)(void*)>(
      vptr[slot])(inst);
}

// OverlayB declared through the interpreter so the overlay runs against
// a reflected, JIT-constructed object -- the language-binding setting.
// Itanium has a destructor pair (D1/D0) before the user virtuals; MSVC
// has a single deleting-dtor slot, so the user-virtual indices differ.
Cpp::DeclRef DeclareBase() {
  // -include new: Construct's wrapper uses placement new, so every
  // interpreter compilation must see <new>.
  Cpp::CreateInterpreter({"-include", "new"});
  Cpp::Declare("struct OverlayB {"
               "  virtual ~OverlayB() {}"
               "  virtual int alpha(int x) { return x + 10; }"
               "  virtual int beta(int x) { return x + 20; }"
               "};");
  return Cpp::GetNamed("OverlayB");
}

Cpp::FuncRef Method(Cpp::DeclRef scope, const char* name) {
  std::vector<Cpp::FuncRef> methods;
  Cpp::GetClassMethods(scope, methods);
  for (auto m : methods)
    if (Cpp::GetName(Cpp::DeclRef{m.data}) == name)
      return m;
  return nullptr;
}

// Dispatch through the installed vtable slot. Calling beta() directly in
// this TU would let the compiler devirtualize and bypass the overlay;
// reading the slot tests what the overlay actually installs.
int call_slot(void* inst, int slot, int arg) {
  void** vptr = *reinterpret_cast<void***>(inst);
  return TestUtils::BitCastFn<int (CPPINTEROP_VTABLE_SLOT_CC *)(void*, int)>(
      vptr[slot])(inst, arg);
}

#ifdef _WIN32
constexpr int kAlpha = 1;
constexpr int kBeta = 2;
#else
constexpr int kAlpha = 2;
constexpr int kBeta = 3;
#endif

} // namespace

TEST(VTableOverlay, ReplacesSlotPreservingOthers) {
  auto B = DeclareBase();
  void* inst = Cpp::Construct(B).data;
  ASSERT_NE(inst, nullptr);
  auto ov = Cpp::MakeUniqueVTableOverlay(
      inst, B, {{Method(B, "beta"), MethodAddr(&Repl::negate)}});
  ASSERT_TRUE(ov);
  EXPECT_EQ(call_slot(inst, kBeta, 5), -5);  // overlaid
  EXPECT_EQ(call_slot(inst, kAlpha, 5), 15); // preserved: alpha -> x+10
  ov.reset(); // restore the vptr before freeing the object
  Cpp::Destruct(inst, B);
}

TEST(VTableOverlay, PreservesPrefixAndUnrelatedSlots) {
  auto B = DeclareBase();
  void* inst = Cpp::Construct(B).data;
  ASSERT_NE(inst, nullptr);
  void** aot = *reinterpret_cast<void***>(inst);
  // Prefix slot(s) immediately before the address point: Itanium has
  // offset-to-top at [-2] and type_info at [-1]; MSVC has just the
  // complete-object-locator at [-1].
  void* prefix_m1 = aot[-1];
#ifndef _WIN32
  void* prefix_m2 = aot[-2];
#endif
  void* alpha_slot = aot[kAlpha];
  auto ov = Cpp::MakeUniqueVTableOverlay(
      inst, B, {{Method(B, "beta"), MethodAddr(&Repl::negate)}});
  ASSERT_TRUE(ov);
  void** now = *reinterpret_cast<void***>(inst);
  EXPECT_EQ(now[-1], prefix_m1);
#ifndef _WIN32
  EXPECT_EQ(now[-2], prefix_m2);
#endif
  EXPECT_EQ(now[kAlpha], alpha_slot); // unrelated slot copied verbatim
  ov.reset();
  Cpp::Destruct(inst, B);
}

TEST(VTableOverlay, RestoresOnDestroy) {
  auto B = DeclareBase();
  void* inst = Cpp::Construct(B).data;
  ASSERT_NE(inst, nullptr);
  void* aot = *reinterpret_cast<void**>(inst);
  {
    auto ov = Cpp::MakeUniqueVTableOverlay(
        inst, B, {{Method(B, "beta"), MethodAddr(&Repl::negate)}});
    ASSERT_TRUE(ov);
    EXPECT_NE(*reinterpret_cast<void**>(inst), aot);
  }
  EXPECT_EQ(*reinterpret_cast<void**>(inst), aot);
  EXPECT_EQ(call_slot(inst, kBeta, 5), 25); // original beta restored: x+20
  Cpp::Destruct(inst, B); // overlay already released by the inner scope
}

TEST(VTableOverlay, ReplacesMultipleSlots) {
  auto B = DeclareBase();
  void* inst = Cpp::Construct(B).data;
  ASSERT_NE(inst, nullptr);
  auto ov = Cpp::MakeUniqueVTableOverlay(
      inst, B,
      {{Method(B, "alpha"), MethodAddr(&Repl::negate)},
       {Method(B, "beta"), MethodAddr(&Repl::twice)}});
  ASSERT_TRUE(ov);
  EXPECT_EQ(call_slot(inst, kAlpha, 5), -5);
  EXPECT_EQ(call_slot(inst, kBeta, 5), 10);
  ov.reset();
  Cpp::Destruct(inst, B);
}

TEST(VTableOverlay, RejectsInvalidInput) {
  auto B = DeclareBase();
  void* inst = Cpp::Construct(B).data;
  ASSERT_NE(inst, nullptr);
  Cpp::ConstFuncRef beta = Method(B, "beta");
  Cpp::ConstFuncRef none = nullptr;
  void* fn = MethodAddr(&Repl::negate);

  EXPECT_EQ(Cpp::MakeVTableOverlay(nullptr, B, &beta, &fn, 1), nullptr); // inst
  EXPECT_EQ(Cpp::MakeVTableOverlay(inst, nullptr, &beta, &fn, 1),
            nullptr);                                                 // base
  EXPECT_EQ(Cpp::MakeVTableOverlay(inst, B, &none, &fn, 1), nullptr); // method
  Cpp::Destruct(inst, B);
}

// Sibling instance of the same type is unaffected when another instance is
// overlaid -- proves per-instance scope of the vptr swap (not per-class).
TEST(VTableOverlay, OverlayIsPerInstance) {
  auto B = DeclareBase();
  void* a = Cpp::Construct(B).data;
  void* b = Cpp::Construct(B).data;
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  void* b_vptr_before = *reinterpret_cast<void**>(b);

  auto ov = Cpp::MakeUniqueVTableOverlay(
      a, B, {{Method(B, "beta"), MethodAddr(&Repl::negate)}});
  ASSERT_TRUE(ov);

  EXPECT_NE(*reinterpret_cast<void**>(a), b_vptr_before); // a swapped
  EXPECT_EQ(*reinterpret_cast<void**>(b), b_vptr_before); // b untouched
  EXPECT_EQ(call_slot(a, kBeta, 5), -5);  // overlay on a
  EXPECT_EQ(call_slot(b, kBeta, 5), 25);  // original on b

  ov.reset();
  Cpp::Destruct(a, B);
  Cpp::Destruct(b, B);
}

// Thunks receive the unmodified `this` pointer of the overlaid instance, so
// they can read the object's data members directly. Layout assumed by the
// test: [ vptr ][ int value ] (true on every ABI we build for).
TEST(VTableOverlay, ThunkReadsThisAndDataMember) {
  Cpp::CreateInterpreter({"-include", "new"});
  Cpp::Declare("struct OverlayWithData {"
               "  int value;"
               "  OverlayWithData() : value(100) {}"
               "  virtual ~OverlayWithData() {}"
               "  virtual int frob(int x) { return x; }"
               "};");
  auto T = Cpp::GetNamed("OverlayWithData");
  ASSERT_NE(T, nullptr);
  void* inst = Cpp::Construct(T).data;
  ASSERT_NE(inst, nullptr);

  auto ov = Cpp::MakeUniqueVTableOverlay(
      inst, T, {{Method(T, "frob"), MethodAddr(&Repl::read_value)}});
  ASSERT_TRUE(ov);

  // First user virtual lives at kAlpha (Itanium D1/D0 prefix; MSVC single
  // deleting-dtor) -- the same layout the existing tests use.
  EXPECT_EQ(call_slot(inst, kAlpha, 5), 105); // value(=100) + x(=5)
  ov.reset();
  Cpp::Destruct(inst, T);
}

// Derived class with an override has the same single-vtable layout as the
// base; overlay still finds and replaces the (overridden) slot.
TEST(VTableOverlay, DerivedClassWithOverride) {
  Cpp::CreateInterpreter({"-include", "new"});
  Cpp::Declare("struct DvBase {"
               "  virtual ~DvBase() {}"
               "  virtual int frob(int x) { return x + 1; }"
               "};"
               "struct DvDerived : DvBase {"
               "  int frob(int x) override { return x + 2; }"
               "};");
  auto D = Cpp::GetNamed("DvDerived");
  ASSERT_NE(D, nullptr);
  void* inst = Cpp::Construct(D).data;
  ASSERT_NE(inst, nullptr);

  // Sanity-check the derived's original slot dispatches to DvDerived::frob.
  EXPECT_EQ(call_slot(inst, kAlpha, 5), 7);

  auto ov = Cpp::MakeUniqueVTableOverlay(
      inst, D, {{Method(D, "frob"), MethodAddr(&Repl::negate)}});
  ASSERT_TRUE(ov);
  EXPECT_EQ(call_slot(inst, kAlpha, 5), -5);
  ov.reset();
  Cpp::Destruct(inst, D);
}

// Three-level single-inheritance chain (A <- B <- C). Slot layout is still
// flat (one vptr), so overlay on C through C's scope replaces the slot.
TEST(VTableOverlay, MultiLevelInheritance) {
  Cpp::CreateInterpreter({"-include", "new"});
  Cpp::Declare("struct MlA { virtual ~MlA() {} virtual int frob(int x) { return x + 1; } };"
               "struct MlB : MlA { int frob(int x) override { return x + 2; } };"
               "struct MlC : MlB { int frob(int x) override { return x + 3; } };");
  auto C = Cpp::GetNamed("MlC");
  ASSERT_NE(C, nullptr);
  void* inst = Cpp::Construct(C).data;
  ASSERT_NE(inst, nullptr);

  EXPECT_EQ(call_slot(inst, kAlpha, 5), 8); // MlC::frob: x+3

  auto ov = Cpp::MakeUniqueVTableOverlay(
      inst, C, {{Method(C, "frob"), MethodAddr(&Repl::twice)}});
  ASSERT_TRUE(ov);
  EXPECT_EQ(call_slot(inst, kAlpha, 5), 10); // overlay: x*2
  ov.reset();
  Cpp::Destruct(inst, C);
}

// Multiple inheritance with two polymorphic direct bases produces two
// vptrs (one per non-empty base subobject); the primary-vptr-only overlay
// cannot retarget the secondary-base dispatch path. MakeVTableOverlay
// refuses such a layout outright so the caller never gets back a handle
// that would mis-dispatch through the untouched secondary vptr.
TEST(VTableOverlay, RejectsMultipleInheritance) {
  Cpp::CreateInterpreter({"-include", "new"});
  Cpp::Declare("struct MiA { virtual ~MiA() {} virtual int af(int x) { return x + 10; } };"
               "struct MiB { virtual ~MiB() {} virtual int bf(int x) { return x + 20; } };"
               "struct MiC : MiA, MiB {"
               "  int af(int x) override { return x + 11; }"
               "  int bf(int x) override { return x + 22; }"
               "};");
  auto C = Cpp::GetNamed("MiC");
  ASSERT_NE(C, nullptr);
  void* inst = Cpp::Construct(C).data;
  ASSERT_NE(inst, nullptr);

  auto ov = Cpp::MakeUniqueVTableOverlay(
      inst, C, {{Method(C, "af"), MethodAddr(&Repl::negate)}});
  EXPECT_FALSE(ov); // refuses the layout

  Cpp::Destruct(inst, C);
}

// Non-polymorphic class has no vtable to overlay; rejected at the
// RD->isPolymorphic() gate inside MakeVTableOverlay.
TEST(VTableOverlay, RejectsNonPolymorphicBase) {
  Cpp::CreateInterpreter({"-include", "new"});
  Cpp::Declare("struct NonPoly { int x; };"
               "struct PolyMethodHolder {"
               "  virtual int dummy(int x) { return x; }"
               "};");
  auto NP = Cpp::GetNamed("NonPoly");
  auto PH = Cpp::GetNamed("PolyMethodHolder");
  ASSERT_NE(NP, nullptr);
  ASSERT_NE(PH, nullptr);
  void* inst = Cpp::Construct(NP).data;
  ASSERT_NE(inst, nullptr);
  Cpp::ConstFuncRef dummy = Method(PH, "dummy");
  void* fn = MethodAddr(&Repl::negate);
  EXPECT_EQ(Cpp::MakeVTableOverlay(inst, NP, &dummy, &fn, 1), nullptr);
  Cpp::Destruct(inst, NP);
}

// A virtual method from an unrelated, larger class has a slot index that
// exceeds the target base's vtable size; applyVTableOverlay's per-slot
// bounds check rejects rather than writing past the overlay block.
TEST(VTableOverlay, RejectsOutOfRangeMethodSlot) {
  Cpp::CreateInterpreter({"-include", "new"});
  Cpp::Declare("struct SmallBase {"
               "  virtual ~SmallBase() {}"
               "  virtual int sf(int x) { return x; }"
               "};"
               "struct LargeUnrelated {"
               "  virtual ~LargeUnrelated() {}"
               "  virtual int v1(int) { return 0; }"
               "  virtual int v2(int) { return 0; }"
               "  virtual int v3(int) { return 0; }"
               "  virtual int v4(int) { return 0; }"
               "  virtual int v5(int) { return 0; }"
               "  virtual int v6(int) { return 0; }"
               "  virtual int v7(int) { return 0; }"
               "  virtual int v8(int) { return 0; }"
               "  virtual int v9(int) { return 0; }"
               "  virtual int v10(int) { return 0; }"
               "};");
  auto Small = Cpp::GetNamed("SmallBase");
  auto Large = Cpp::GetNamed("LargeUnrelated");
  ASSERT_NE(Small, nullptr);
  ASSERT_NE(Large, nullptr);
  void* inst = Cpp::Construct(Small).data;
  ASSERT_NE(inst, nullptr);
  Cpp::ConstFuncRef v10 = Method(Large, "v10");
  void* fn = MethodAddr(&Repl::negate);
  EXPECT_EQ(Cpp::MakeVTableOverlay(inst, Small, &v10, &fn, 1), nullptr);
  Cpp::Destruct(inst, Small);
}

// Overlay across a A <- B hierarchy using replacements that read the
// object's data members through typed static_cast on `self`. Objects are
// stack-allocated by the test TU (matching the interpreter-side layout the
// reflection API computes the slot indices for) and initialised with
// non-zero member values; the replacements pull those values back out and
// fold them into the return, exercising the live-`this` path end to end.
TEST(VTableOverlay, OverlayThroughHierarchyAccessesDataMembers) {
  Cpp::CreateInterpreter({});
  Cpp::Declare("struct A {"
               "  int m_x;"
               "  A(int x) : m_x(x) {}"
               "  virtual ~A() {}"
               "  virtual int method() { return m_x; }"
               "};"
               "struct B : A {"
               "  double m_d;"
               "  B(int x, double d) : A(x), m_d(d) {}"
               "  int method() override { return (int)m_d; }"
               "};");
  auto A_scope = Cpp::GetNamed("A");
  auto B_scope = Cpp::GetNamed("B");
  ASSERT_NE(A_scope, nullptr);
  ASSERT_NE(B_scope, nullptr);

  A a(5);
  B b(7, 3.5);

  // Baseline: the host-compiler-built objects dispatch to their C++ bodies.
  EXPECT_EQ(call_slot_no_arg(&a, kAlpha), 5);  // A::method returns m_x
  EXPECT_EQ(call_slot_no_arg(&b, kAlpha), 3);  // B::method returns (int)m_d

  auto ov_a = Cpp::MakeUniqueVTableOverlay(
      &a, A_scope,
      {{Method(A_scope, "method"), MethodAddr(&Repl::foo)}});
  ASSERT_TRUE(ov_a);
  auto ov_b = Cpp::MakeUniqueVTableOverlay(
      &b, B_scope,
      {{Method(B_scope, "method"), MethodAddr(&Repl::bar)}});
  ASSERT_TRUE(ov_b);

  EXPECT_EQ(call_slot_no_arg(&a, kAlpha), 15); // foo: A::m_x(5) + 10
  EXPECT_EQ(call_slot_no_arg(&b, kAlpha), 10); // bar: (int)(7 + 3.5)

  ov_a.reset();
  ov_b.reset();
  // No Destruct: stack-allocated objects, C++ dtors fire on scope exit
  // (vptrs restored above so virtual dispatch lands on the real dtor).
}

// Virtual inheritance has a longer pre-address-point prefix (vbase-offset
// entries) and a vtable-in-vbase that carries `_ZTv0_n*` virtual thunks for
// dispatch through the virtual-base pointer -- neither is covered by the
// primary-vptr overlay. MakeVTableOverlay refuses, mirroring the multi-
// inheritance case. Reviewer ref: stackoverflow.com/a/39182009.
TEST(VTableOverlay, RejectsVirtualInheritance) {
  Cpp::CreateInterpreter({"-include", "new"});
  Cpp::Declare("struct ViBase {"
               "  virtual ~ViBase() {}"
               "  virtual int frob(int x) { return x + 1; }"
               "};"
               "struct ViDerived : virtual ViBase {"
               "  int frob(int x) override { return x + 2; }"
               "};");
  auto D = Cpp::GetNamed("ViDerived");
  ASSERT_NE(D, nullptr);
  void* inst = Cpp::Construct(D).data;
  ASSERT_NE(inst, nullptr);

  auto ov = Cpp::MakeUniqueVTableOverlay(
      inst, D, {{Method(D, "frob"), MethodAddr(&Repl::negate)}});
  EXPECT_FALSE(ov); // refuses the layout

  Cpp::Destruct(inst, D);
}

// on_destroy fires once on the deleting-dtor path (before the original
// destructor); receives `inst` and `cleanup_data` verbatim.
TEST(VTableOverlay, DestructorHookFires) {
  auto B = DeclareBase();
  void* inst = Cpp::Construct(B).data;
  ASSERT_NE(inst, nullptr);

  struct State { int fires = 0; void* last_inst = nullptr; };
  State state;
  auto* ov = Cpp::MakeVTableOverlay(
      inst, B, /*methods=*/nullptr, /*overlay_fns=*/nullptr,
      /*n_overlays=*/0, /*n_extra_prefix_slots=*/0,
      /*on_destroy=*/
      [](void* i, void* data) {
        auto* s = static_cast<State*>(data);
        s->fires += 1;
        s->last_inst = i;
      },
      /*cleanup_data=*/&state);
  ASSERT_NE(ov, nullptr);
  EXPECT_EQ(state.fires, 0); // not fired yet

  Cpp::Destruct(inst, B); // runs the wrapped deleting dtor

  EXPECT_EQ(state.fires, 1);
  EXPECT_EQ(state.last_inst, inst);

  // Overlay handle outlives the instance: DestroyVTableOverlay must
  // skip the vptr restore (instance freed) but still free the block.
  Cpp::DestroyVTableOverlay(ov);
}

// Two overlays installed via different interpreters: each carries its
// own hook state on its own block, so destruction of one does not fire
// the other's callback. Routing is per-instance (via the hidden
// self-pointer slot in each block), not per-interpreter.
TEST(VTableOverlay, DestructorHookIsPerInstance) {
  Cpp::InterpRef I1 = Cpp::CreateInterpreter({"-include", "new"});
  ASSERT_NE(I1, nullptr);
  Cpp::Declare("struct DhBase1 { virtual ~DhBase1() {} "
               "virtual int frob(int x) { return x + 1; } };");
  auto B1 = Cpp::GetNamed("DhBase1");
  ASSERT_NE(B1, nullptr);
  void* inst1 = Cpp::Construct(B1).data;
  ASSERT_NE(inst1, nullptr);
  int counter1 = 0;
  auto* ov1 = Cpp::MakeVTableOverlay(
      inst1, B1, nullptr, nullptr, 0, 0,
      [](void* /*i*/, void* data) { *static_cast<int*>(data) += 1; },
      &counter1);
  ASSERT_NE(ov1, nullptr);

  Cpp::InterpRef I2 = Cpp::CreateInterpreter({"-include", "new"});
  ASSERT_NE(I2, nullptr);
  Cpp::Declare("struct DhBase2 { virtual ~DhBase2() {} "
               "virtual int frob(int x) { return x + 2; } };");
  auto B2 = Cpp::GetNamed("DhBase2");
  ASSERT_NE(B2, nullptr);
  void* inst2 = Cpp::Construct(B2).data;
  ASSERT_NE(inst2, nullptr);
  int counter2 = 0;
  auto* ov2 = Cpp::MakeVTableOverlay(
      inst2, B2, nullptr, nullptr, 0, 0,
      [](void* /*i*/, void* data) { *static_cast<int*>(data) += 1; },
      &counter2);
  ASSERT_NE(ov2, nullptr);

  // Re-activate I1 so Cpp::Destruct's reflection lookup runs against
  // the scope's owning interpreter; only counter1 must fire.
  Cpp::ActivateInterpreter(I1);
  Cpp::Destruct(inst1, B1);
  EXPECT_EQ(counter1, 1);
  EXPECT_EQ(counter2, 0);

  Cpp::ActivateInterpreter(I2);
  Cpp::Destruct(inst2, B2);
  EXPECT_EQ(counter1, 1);
  EXPECT_EQ(counter2, 1);

  Cpp::DestroyVTableOverlay(ov1);
  Cpp::DestroyVTableOverlay(ov2);
}

// on_destroy = nullptr is the "opt-in, zero overhead" contract: no
// wrapper is installed, so the deleting-dtor slot in the overlay block
// is a verbatim copy of the original. Pin that directly rather than
// inferring it from other tests passing.
TEST(VTableOverlay, DestructorHookOptInZero) {
  auto B = DeclareBase();
  void* inst = Cpp::Construct(B).data;
  ASSERT_NE(inst, nullptr);
  void** orig_vptr = *reinterpret_cast<void***>(inst);
#ifdef _WIN32
  constexpr int kDDtor = 0;
#else
  constexpr int kDDtor = 1;
#endif
  void* orig_d = orig_vptr[kDDtor];

  auto* ov = Cpp::MakeVTableOverlay(
      inst, B, nullptr, nullptr, 0, 0,
      /*on_destroy=*/nullptr, /*cleanup_data=*/nullptr);
  ASSERT_NE(ov, nullptr);

  void** new_vptr = *reinterpret_cast<void***>(inst);
  EXPECT_EQ(new_vptr[kDDtor], orig_d);

  Cpp::DestroyVTableOverlay(ov);
  Cpp::Destruct(inst, B);
}

// Caller-driven teardown before the C++ destruction restores the
// original vptr in ~VTableOverlay, so the wrapper is no longer
// installed and Cpp::Destruct calls the unhooked deleting dtor; the
// callback does not fire.
TEST(VTableOverlay, DestructorHookSkippedOnCallerTeardown) {
  auto B = DeclareBase();
  void* inst = Cpp::Construct(B).data;
  ASSERT_NE(inst, nullptr);

  int fires = 0;
  auto* ov = Cpp::MakeVTableOverlay(
      inst, B, nullptr, nullptr, 0, 0,
      [](void* /*i*/, void* data) { *static_cast<int*>(data) += 1; }, &fires);
  ASSERT_NE(ov, nullptr);

  Cpp::DestroyVTableOverlay(ov);
  Cpp::Destruct(inst, B);
  EXPECT_EQ(fires, 0);
}

// Phase 1 extra-prefix slots: a binding stashes per-instance data in slots
// immediately before the ABI prefix; thunks read it via the inline
// VTableOverlayExtraSlot helper -- a single fixed-offset load.
TEST(VTableOverlay, SetsExtraPrefixSlots) {
  auto B = DeclareBase();
  void* inst = Cpp::Construct(B).data;
  ASSERT_NE(inst, nullptr);
  auto ov = Cpp::MakeUniqueVTableOverlay(
      inst, B, {{Method(B, "beta"), MethodAddr(&Repl::negate)}},
      /*n_extra_prefix_slots=*/2);
  ASSERT_TRUE(ov);
  void*& slot0 = Cpp::VTableOverlayExtraSlot(inst, 0);
  void*& slot1 = Cpp::VTableOverlayExtraSlot(inst, 1);
  EXPECT_EQ(slot0, nullptr); // nullptr-initialized
  EXPECT_EQ(slot1, nullptr);
  slot0 = reinterpret_cast<void*>(uintptr_t{0xC0FFEE});
  slot1 = reinterpret_cast<void*>(uintptr_t{0xDEADBEEF});
  EXPECT_EQ(slot0, reinterpret_cast<void*>(uintptr_t{0xC0FFEE}));
  EXPECT_EQ(slot1, reinterpret_cast<void*>(uintptr_t{0xDEADBEEF}));
  // The overlaid slot still dispatches normally; extra-slot data is opaque.
  EXPECT_EQ(call_slot(inst, kBeta, 5), -5);
  ov.reset();
  Cpp::Destruct(inst, B);
}
