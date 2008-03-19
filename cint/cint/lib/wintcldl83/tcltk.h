/* /% C %/ */
/***********************************************************************
 * The WildCard interpreter
 ************************************************************************
 * parameter information file TCLTK.h
 ************************************************************************
 * Description:
 *  This header file is given to makecint by -h option.  This header file
 *  is parsed by C preprocessor then by cint.
 ************************************************************************
 * Copyright(c) 1996-1997  Masaharu Goto (MXJ02154@niftyserve.or.jp)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef G__TCLTK_H
#define G__TCLTK_H

#ifdef __MAKECINT__
/**************************************************************
* Defining _XLIB_H here prevents reading X11 header files in
* tk.h. There are several X11 declarations needed in tk.h though,
* the dummy is provided here.
**************************************************************/
#define _XLIB_H
#ifndef __STDC__
#define __STDC__
#endif
typedef int Bool ;
typedef struct { } Visual; /* ./tk.h 380 */
typedef unsigned long Window ; /* ./tk.h 382 */
typedef struct { } XWindowChanges ; /* ./tk.h 391 */
typedef struct { } XSetWindowAttributes ; /* ./tk.h 393 */
typedef struct { } XColor ;
typedef struct { } XFontStruct ;
typedef unsigned long Pixmap ;
typedef unsigned long Cursor ;
typedef unsigned long Colormap;
typedef unsigned long Atom ;
typedef union _XEvent { } XEvent ;
typedef struct {} XPoint ;
typedef struct {} XGCValues ;
typedef struct _XGC { } *GC ;
typedef struct _XDisplay { } Display; /* ./tk.h 377 */

/* tcl8.3.2 */
typedef void* Tcl_ThreadCreateProc;
typedef long Time;
typedef void* Drawable;
typedef int XID;
typedef long Font;

#endif /* __MAKECINT__ */

/**************************************************************
* include tk.h
**************************************************************/
#define XLIB_ILLEGAL_ACCESS
#include <tk.h>

/**************************************************************
* global interpreter object instantiated by Tk in TkInit.c is
* exposed to CINT.
**************************************************************/
extern Tcl_Interp *interp;

/**************************************************************
* add #pragma tcl statement in the CINT parser
**************************************************************/
#ifndef __MAKECINT__
void G__cinttk_init();
#endif

/**************************************************************
* WildCard/X11 Event Loop function
**************************************************************/
void WildCard_MainLoop();
void WildCard_Exit();
int WildCard_AllocConsole();
int WildCard_FreeConsole();

#ifdef __MAKECINT__
/**************************************************************
* following description is optional, hence, cint -c-2
* automatically ignores undefined structs for creating
* interface method.
**************************************************************/
#pragma link off class Tcl_AsyncHandler_;
#pragma link off class Tcl_RegExp_;
#pragma link off class Tk_BindingTable_;
#pragma link off class Tk_Canvas_;
#pragma link off class Tk_ErrorHandler_;
#pragma link off class Tk_Image__;
#pragma link off class Tk_ImageMaster_;
#pragma link off class Tk_TimerToken_;
#pragma link off class Tk_Window_;
#pragma link off class Tk_3DBorder_;
#pragma link off class _XGC;
#pragma link off class _XEvent;
#pragma link off class _XDisplay;
#pragma link off class $fpos_t;
#pragma link off function Tcl_CreatePipeline;
#pragma link off function Tcl_AppInit;
#pragma link off function Tcl_GetCwd;
#pragma link off function Tcl_GetOpenFile;
#pragma link off function Tcl_MakeTcpClientChannel;
#pragma link off function Tcl_DumpActiveMemory;
#pragma link off function Tk_FileeventCmd;
#ifdef __hpux
#pragma link off function Tk_ConfigureFree;
#pragma link off function Tk_ApplicationCmd;
#endif
#endif /* __MAKECINT__ */

#endif /* G__TCLTK_H */

/*
 * Local Variables:
 * c-tab-always-indent:nil
 * c-indent-level:2
 * c-continued-statement-offset:2
 * c-brace-offset:-2
 * c-brace-imaginary-offset:0
 * c-argdecl-indent:0
 * c-label-offset:-2
 * compile-command:"make -k"
 * End:
 */
