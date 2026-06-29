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

#endif // UNITTESTS_CPPINTEROP_TESTSHAREDLIB_TESTSHAREDLIB_H
