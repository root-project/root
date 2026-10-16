#ifndef CPYRT_COMMONDEFS_H
#define CPYRT_COMMONDEFS_H

// export macros for our own API
// import/export (after precommondefs.h from PyPy)
#ifdef _MSC_VER
// Windows requires symbols to be explicitly exported
#define CPYRT_EXPORT extern __declspec(dllexport)
#define CPYRT_IMPORT extern __declspec(dllimport)
#define CPYRT_CLASS_EXPORT __declspec(dllexport)

// CPYRT_EXTERN is dual use in the public API
#ifndef CPYRT_INTERNAL
#define CPYRT_EXTERN extern __declspec(dllexport)
#define CPYRT_CLASS_EXTERN __declspec(dllexport)
#else
#define CPYRT_EXTERN extern __declspec(dllimport)
#define CPYRT_CLASS_EXTERN __declspec(dllimport)
#endif

#define CPYRT_STATIC

#else
// Linux, Mac, etc.
#define CPYRT_EXPORT extern
#define CPYRT_IMPORT extern
#define CPYRT_CLASS_EXPORT
#define CPYRT_EXTERN extern
#define CPYRT_CLASS_EXTERN
#define CPYRT_STATIC static

#endif

// How JIT-compiled callback wrappers (function pointers, std::function, and
// virtual overrides implemented in Python) report a failed Python call. The
// default is to throw a cpyrt::PyException from the wrapper, which unwinds
// through the C++ frames in between (JIT-compiled, or built by another
// compiler) up to the try block in CPPMethod::ExecuteFast. That unwind hangs
// on Windows, and on Apple ARM64 exceptions through JIT-compiled frames are
// known not to work (the test suite skips the tests that catch a PyException
// in C++ there) and are the prime suspect for the sporadic crash of
// test_datatypes on the macOS beta CI runner. On those platforms the wrapper
// therefore leaves the Python error set and returns a default value instead;
// the C++ code keeps running and ExecuteFast turns the pending error into a
// failed call once the C++ call returns. The price is that C++ code can not
// catch the Python error as a PyException.
#if defined(_WIN32) || (defined(__APPLE__) && defined(__aarch64__))
#define CPYRT_CALLBACK_NOTHROW 1
#else
#define CPYRT_CALLBACK_NOTHROW 0
#endif

#endif // !CPYRT_COMMONDEFS_H
