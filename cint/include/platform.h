/***************************************************************************
* include/platform.h
*
*  eliminate platform dependent symbol
* this include file must be used with -p or +P,-P command line option for
* two-path processing with external C/C++ preprocessor
***************************************************************************/

#ifndef G__PLATFORM_H
#define G__PLATFORM_H

#ifdef __CINT__

/**************************************************************************
* windows
***************************************************************************/
#define __stdcall
#define __cdecl
#define _cdecl
#define __far
#define __huge
#define __declspec(x)

typedef void* LPCMENUITEMINFOA;
struct IRpcStubBuffer;
struct IRpcChannelBuffer;

/* added */
typedef double __int64;
typedef double DWORDLONG;
#define NTAPI
#define WINAPI
#define WINGDIAPI
#define APIENTRY

/* VC++5.0 */
#define MIDL_PASS
struct _SID { };
typedef unsigned long ULONGLONG;
typedef double LONGLONG;
typedef void VOID;
typedef int INT;

/* STL */
#define _THROW_NONE

/**************************************************************************
* gcc
***************************************************************************/
#ifdef G__FBSD
#include <time.h>
#endif

#ifndef G__FBSD
#define __signed__ 
#define __const
#endif

//#define __BEGIN_DECLS
#define __extension__

/**************************************************************************
* KAI C++ compiler
***************************************************************************/
typedef short __wchar_t;

/**************************************************************************
* etc
***************************************************************************/
#define _const
#define _signed 
#define __signed 

/**************************************************************************
* Gunay ARSLEN's contribution
***************************************************************************/
#define __const__

#define __inline__
#define __inline

typedef long __kernel_loff_t;  /* must be long long */
typedef unsigned long __u64;
typedef long __s64;


#endif /* __CINT__ */
#endif /* G__PLATFORM_H */
