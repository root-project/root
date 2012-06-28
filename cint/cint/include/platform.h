/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
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
typedef long long __int64;
typedef double DWORDLONG;
#ifndef NTAPI
#define NTAPI
#endif
#ifndef WINAPI
#define WINAPI
#endif
#ifndef WINGDIAPI
#define WINGDIAPI
#endif
#ifndef APIENTRY
#define APIENTRY
#endif 

/* VC++8.0 */
typedef unsigned int *PUINT_PTR;
typedef unsigned long long ULONG64;

/* VC++7.0 */
#if defined(_MSC_VER) && (_MSC_VER>=1300) /* block for VC7, more refinment? */
typedef DWORDLONG __w64;
typedef unsigned int UINT_PTR;
typedef void** HANDLE_PTR;
#endif

/* VC++5.0 */
#if defined(_MSC_VER) && (_MSC_VER<1300) /* block for VC7, more refinment? */
#define MIDL_PASS
#endif
struct _SID { };
typedef unsigned long ULONGLONG;
typedef double LONGLONG;
typedef void VOID;
typedef int INT;

/* Borland C++ compiler 5.5 */
typedef char CHAR;
typedef short WCHAR;
typedef long DWORD;
typedef double DWORD64;
typedef unsigned int UINT;
typedef unsigned long* ULONG_PTR;
typedef double* PULONG_PTR;
typedef long LONG;
typedef struct tagPOINT {
    LONG x;
    LONG y;
} POINT;
typedef struct tagPOINT *LPPOINT;
typedef void* HWND;
typedef struct tagMOUSEHOOKSTRUCT {
    POINT   pt;
    HWND    hwnd;
    UINT    wHitTestCode;
    ULONG_PTR dwExtraInfo;
} MOUSEHOOKSTRUCT, *LPMOUSEHOOKSTRUCT, *PMOUSEHOOKSTRUCT;
typedef struct tagMOUSEHOOKSTRUCTEX
{
    MOUSEHOOKSTRUCT dmy;
    DWORD   mouseData;
} MOUSEHOOKSTRUCTEX, *LPMOUSEHOOKSTRUCTEX, *PMOUSEHOOKSTRUCTEX;
typedef struct tagRECT
    {
    LONG left;
    LONG top;
    LONG right;
    LONG bottom;
    }	RECT;
typedef struct tagRECT *LPRECT;
typedef struct tagMONITORINFO
{
    DWORD   cbSize;
    RECT    rcMonitor;
    RECT    rcWork;
    DWORD   dwFlags;
} MONITORINFO, *LPMONITORINFO;
#define CCHDEVICENAME 32
typedef struct tagMONITORINFOEXA
{
    MONITORINFO dmy;
    CHAR        szDevice[CCHDEVICENAME];
} MONITORINFOEXA, *LPMONITORINFOEXA;
typedef struct tagMONITORINFOEXW
{
    MONITORINFO dmy;
    WCHAR       szDevice[CCHDEVICENAME];
} MONITORINFOEXW, *LPMONITORINFOEXW;
typedef struct { } CHOOSECOLOR, *LPCHOOSECOLOR;
typedef struct { } CHOOSEFONT, *LPCHOOSEFONT;
typedef struct { } OPENFILENAME, *LPOPENFILENAME;
typedef struct { } PAGESETUPDLG, *LPPAGESETUPDLG;
typedef struct { } FINDREPLACE, *LPFINDREPLACE;
typedef struct { } PRINTDLG, *LPPRINTDLG;

/* STL */
#define _THROW_NONE

/**************************************************************************
* gcc
***************************************************************************/
#ifdef G__FBSD
#include <time.h>
#endif

#ifdef G__OBSD
#include <time.h>
#endif

#ifndef G__FBSD
#define __signed__ 
#define __const
#endif

//#define __BEGIN_DECLS
#define __extension__
#define  __attribute__(x)  

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

#define __inline__ inline
#define __inline inline
#define __forceinline inline


typedef long __kernel_loff_t;  /* must be long long */
typedef unsigned long __u64;
typedef long __s64;
#define __ptr64

//#if defined(G__APPLE) && !defined(G__64BIT)
typedef	int int32_t;
typedef int32_t time_t;
//#endif


#endif /* __CINT__ */
#endif /* G__PLATFORM_H */
