#include "CppInterOp/CppInterOpThunks.h"

#include "gtest/gtest.h"

#include <cstddef>

// Cpp::Thunks::dispatch<Traits, Slot, R, Args...> turns a per-call
// virtual dispatch into the binding's Traits::Call. It is consumed in
// two ways: (a) directly as a function-pointer constant, e.g.
//   void* slot = &Cpp::Thunks::dispatch<Traits, Slot, R, Args...>;
// to install into an overlay's vtable slot; (b) called like an ordinary
// function for unit testing. This file exercises (b) end-to-end and
// verifies the Traits contract.

namespace {

// Tiny stand-in for the per-instance binding state. A real binding
// (CPyCppyy's PyVirtualHandler) keeps a richer object here.
struct Handler {
  int Value;
};

// Traits contract: From() locates the binding's Handler from `self`,
// Call() marshals and forwards. The template form matches what
// dispatch instantiates against.
struct AccumTraits {
  using HandlerType = Handler;
  static Handler& From(void* self) { return *static_cast<Handler*>(self); }
  template <class R, class... Args>
  static R Call(Handler& H, std::size_t Slot, Args... args) {
    // Mimic a marshaling kernel: fold all inputs into the return value
    // so each (R, Slot, Args...) instantiation produces a distinguishable
    // result the test can pin. Accumulate in R so the test can exercise
    // non-integral returns without truncation.
    R acc = static_cast<R>(H.Value) + static_cast<R>(Slot);
    ((acc += static_cast<R>(args)), ...);
    return acc;
  }
};

// Void-return / no-Args Traits. Lives at namespace scope because C++17
// forbids member templates inside local classes (which is what
// TEST(...) bodies introduce).
struct VoidTraits {
  using HandlerType = int;
  static int& From(void* self) { return *static_cast<int*>(self); }
  template <class R, class... Args> static R Call(int& H, std::size_t Slot) {
    H += static_cast<int>(Slot) + 1;
  }
};

} // namespace

// dispatch is a function template. Each (Traits, Slot, R, Args...) tuple
// is a distinct instantiation with the right calling convention to be
// installed at a virtual-method slot.
TEST(CppInterOpThunks, DispatchForwardsThroughTraits) {
  Handler H{100};

  // (Slot=0, R=int, Args=int)
  int (*Fn0)(void*, int) = &Cpp::Thunks::dispatch<AccumTraits, 0, int, int>;
  EXPECT_EQ(Fn0(&H, 7), 107); // 100 + 0 + 7

  // (Slot=3, R=long long, Args=int, int) -- exercises a wider signature
  long long (*Fn1)(void*, int, int) =
      &Cpp::Thunks::dispatch<AccumTraits, 3, long long, int, int>;
  EXPECT_EQ(Fn1(&H, 4, 5), 112); // 100 + 3 + 4 + 5

  // (Slot=2, R=double, Args=double) -- non-integral return
  double (*Fn2)(void*, double) =
      &Cpp::Thunks::dispatch<AccumTraits, 2, double, double>;
  EXPECT_DOUBLE_EQ(Fn2(&H, 0.5), 102.5); // 100 + 2 + 0.5
}

// Different (Slot, R, Args...) tuples must yield different function
// pointers; the dispatcher is not folded across slots even when the
// underlying Call is the same template.
TEST(CppInterOpThunks, DistinctInstantiationsHaveDistinctAddresses) {
  void* A =
      reinterpret_cast<void*>(&Cpp::Thunks::dispatch<AccumTraits, 0, int, int>);
  void* B =
      reinterpret_cast<void*>(&Cpp::Thunks::dispatch<AccumTraits, 1, int, int>);
  EXPECT_NE(A, B); // Slot 0 vs Slot 1
}

// Zero-arg case: dispatch with no Args... and a void return mirrors the
// signature of a virtual method `void foo()` -- the minimal shape an
// overlay would install.
TEST(CppInterOpThunks, DispatchWithNoArgsAndVoidReturn) {
  int H = 0;
  void (*Fn)(void*) = &Cpp::Thunks::dispatch<VoidTraits, 5, void>;
  Fn(&H);
  EXPECT_EQ(H, 6); // 0 + 5 + 1
}
