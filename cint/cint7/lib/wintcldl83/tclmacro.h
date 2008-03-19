/* /% C %/ */
/***********************************************************************
 * The WildCard interpreter
 ************************************************************************
 * parameter information file tclmacro.h
 ************************************************************************
 * Description:
 *  Constant macro and function macro to be exposed to C/C++ interpreter.
 ************************************************************************
 * Copyright(c) 1996-1997  Masaharu Goto (MXJ02154@niftyserve.or.jp)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#ifndef _TCL
#define _TCL

#define TCL_MAJOR_VERSION 7
#define TCL_MINOR_VERSION 4

#define TCL_OK		0
#define TCL_ERROR	1
#define TCL_RETURN	2
#define TCL_BREAK	3
#define TCL_CONTINUE	4

#define TCL_RESULT_SIZE 200

#define TCL_DSTRING_STATIC_SIZE 200

int Tcl_DStringLength(Tcl_DString* dsPtr);
char *Tcl_DStringValue(Tcl_DString* dsPtr);
void Tcl_DStringTrunc(Tcl_DString *dsPtr,int length);

#define TCL_MAX_PREC 17
#define TCL_DOUBLE_SPACE (TCL_MAX_PREC+10)

#define TCL_DONT_USE_BRACES	1

#define TCL_NO_EVAL		0x10000
#define TCL_EVAL_GLOBAL		0x20000

#define TCL_VOLATILE ((Tcl_FreeProc *)1)
#define TCL_STATIC   ((Tcl_FreeProc *) 0)
#define TCL_DYNAMIC  ((Tcl_FreeProc *) 3)

#define TCL_GLOBAL_ONLY		1
#define TCL_APPEND_VALUE	2
#define TCL_LIST_ELEMENT	4
#define TCL_TRACE_READS		0x10
#define TCL_TRACE_WRITES	0x20
#define TCL_TRACE_UNSETS	0x40
#define TCL_TRACE_DESTROYED	0x80
#define TCL_INTERP_DESTROYED	0x100
#define TCL_LEAVE_ERR_MSG	0x200

#define TCL_LINK_INT		1
#define TCL_LINK_DOUBLE		2
#define TCL_LINK_BOOLEAN	3
#define TCL_LINK_STRING		4
#define TCL_LINK_READ_ONLY	0x80

#define TCL_FILE_READABLE	1
#define TCL_FILE_WRITABLE	2

void Tcl_FreeResult(Tcl_Interp* interp);

#define TCL_SMALL_HASH_TABLE 4

#define TCL_STRING_KEYS		0
#define TCL_ONE_WORD_KEYS	1

ClientData Tcl_GetHashValue(Tcl_HashEntry* h);
void Tcl_SetHashValue(Tcl_HashEntry* h,long value);
char* Tcl_GetHashKey(Tcl_HashTable* tablePtr,Tcl_HashEntry* h);

char* Tcl_FindHashEntry(Tcl_HashTable* tablePtr,char *key);
void Tcl_CreateHashEntry(Tcl_HashTable* tablePtr,char *key,int *newPtr);

void Tcl_Return(Tcl_Interp *interp, char *string, Tcl_FreeProc *freeProc);

#endif /* _TCL */

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
