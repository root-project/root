/* /% C %/ */
/***********************************************************************
 * The WildCard interpreter
 ************************************************************************
 * parameter information file TOP.h
 ************************************************************************
 * Description:
 *  This header file is given to makecint by -h option. The top level
 *  parameter information file.
 ************************************************************************
 * Copyright(c) 1996-1997  Masaharu Goto (MXJ02154@niftyserve.or.jp)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef G__TCLTKTOP_H
#define G__TCLTKTOP_H

/***********************************************************************
* tk.h is included in TCLTK.h. TCLTK.h and tk.h are preprocessed by
* external C preprocessor. All of the macro information is lost in this
* process. 
***********************************************************************/
#ifdef __MAKECINT__
#pragma preprocessor on
#endif

#include "TCLTK.h"

#ifdef __MAKECINT__
#pragma preprocessor off 
#endif


#ifdef __MAKECINT__
/***********************************************************************
* Following part is added to exposed symbols to interpreter. TOP.h is
* not preprocessed. macros included below will be visible from interpreter.
***********************************************************************/
#include "TKMACRO.h"
#include "TCLMACRO.h"

#define SIMPLE
#ifdef SIMPLE
#pragma link off all functions;
#pragma link off all globals;
#pragma link off all classes;

#pragma link C function WildCard_MainLoop;
#pragma link C function WildCard_Exit;
#pragma link C global interp;

#pragma link C function Tcl_Eval;
#pragma link C function Tcl_LinkVar;
#pragma link C function Tcl_UnlinkVar;
#pragma link C function Tcl_SetResult;
#pragma link C function Tcl_EvalFile;
#pragma link C function Tcl_CreateCommand;
#endif

#endif /* __MAKECINT__ */

#endif /* G__TCLTKTOP_H */

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
