#ifndef UNITTESTS_CPPINTEROP_TESTSHAREDLIB_TESTSHAREDLIB_H
#define UNITTESTS_CPPINTEROP_TESTSHAREDLIB_TESTSHAREDLIB_H

#ifdef _WIN32
#define TESTSHAREDLIB_API __declspec(dllexport)
#else
#define TESTSHAREDLIB_API __attribute__((visibility("default")))
#endif

// Avoid having to mangle/demangle the symbol name in tests
extern "C" TESTSHAREDLIB_API int ret_zero();

// A polymorphic type whose vtable is anchored in this shared library,
// plus a dispatch helper compiled here. Calling OverlayDispatchOnce
// from another translation unit is a genuine cross-DSO virtual call
// the caller's compiler cannot devirtualize or inline -- the honest
// setting for measuring VTableOverlay dispatch cost.
struct TESTSHAREDLIB_API OverlayBase {
  OverlayBase();
  virtual ~OverlayBase();
  virtual int frob(int x);
};

extern "C" TESTSHAREDLIB_API int OverlayDispatchOnce(OverlayBase* b, int x);

// Header-*defined* singleton state, in the two shapes that compile to weak
// globals: a function-local static in an inline function and a C++17 inline
// static data member. TestSharedLib.cpp compiles an AOT copy of both (the
// accessors below); jitted code that repeats these definitions must bind the
// library's copies rather than materialize duplicates.
struct TESTSHAREDLIB_API SingletonFixture {
  static SingletonFixture& get() {
    static SingletonFixture instance;
    return instance;
  }

  static inline int s_inline_member = 0;

  int value = 0;
};

extern "C" TESTSHAREDLIB_API void* singleton_fixture_meyers_addr();
extern "C" TESTSHAREDLIB_API int singleton_fixture_meyers_value();
extern "C" TESTSHAREDLIB_API void* singleton_fixture_member_addr();
extern "C" TESTSHAREDLIB_API int singleton_fixture_member_value();

// A *dynamically* initialized header singleton: the local static's initializer
// calls an out-of-line constructor the compiler cannot fold, so clang emits a
// _ZGV<...> guard variable beside the data. bindProcessWeakGlobals must demote
// the guard together with the data -- and only when the process exports both --
// so jitted code neither re-runs initialization nor reads an uninitialized
// instance. The library's accessors init this copy AOT (value 7).
struct TESTSHAREDLIB_API DynamicInitSingleton {
  int value;
  DynamicInitSingleton(); // out-of-line: forces a guarded dynamic init
  static DynamicInitSingleton& get() {
    static DynamicInitSingleton instance;
    return instance;
  }
};

extern "C" TESTSHAREDLIB_API void* dynamic_singleton_addr();
extern "C" TESTSHAREDLIB_API int dynamic_singleton_value();

// A weak global the jitted module forces into @llvm.compiler.used (via its
// __attribute__((used)) definition). Demoting it means it must also be stripped
// from that list -- a declaration there fails the IR verifier -- exercising the
// used-list rewrite. The library exports a strong copy for the JIT to bind.
extern "C" TESTSHAREDLIB_API int g_used_singleton;
extern "C" TESTSHAREDLIB_API void* g_used_singleton_addr();
extern "C" TESTSHAREDLIB_API int g_used_singleton_value();

#endif // UNITTESTS_CPPINTEROP_TESTSHAREDLIB_TESTSHAREDLIB_H
