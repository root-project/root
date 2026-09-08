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

#endif // !CPYRT_COMMONDEFS_H
