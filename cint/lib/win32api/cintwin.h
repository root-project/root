/***********************************************************************
* cintwin.h
*   Embedding Win32 API to CINT
***********************************************************************/

#ifndef G__CINTWIN_H
#define G__CINTWIN_H

/* cint headers */
#ifdef __CINT__
#include <stdlib.h> 
#include <platform.h>
#endif

/* windows header */
#if defined(__MAKECINT__) && (defined(G__BORLAND)||defined(G__VISUAL))
#undef MIDL_PASS
#endif
#include <windows.h>

#ifdef __MAKECINT__
#pragma link off all functions;
#pragma link off all classes;
#pragma link off all globals;
#pragma link off all typedefs;
#endif /* __MAKECINT__ */

#endif
