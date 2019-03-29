#ifndef CPYCPPYY_COMMONDEFS
#define CPYCPPYY_COMMONDEFS

// export macros for our own API
// import/export (after precommondefs.h from PyPy)
#ifdef _MSC_VER
#define CPYCPPYY_EXPORT extern __declspec(dllexport)
#define CPYCPPYY_IMPORT extern __declspec(dllimport)
#define CPYCPPYY_CLASS_EXPORT __declspec(dllexport)
#else
#define CPYCPPYY_EXPORT extern
#define CPYCPPYY_IMPORT extern
#define CPYCPPYY_CLASS_EXPORT
#endif

#endif // !CPYCPPYY_COMMONDEFS
