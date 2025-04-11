/********************************************************************
* mydico.h
********************************************************************/
#ifdef __CINT__
#error mydico.h/C is only for compilation. Abort cint.
#endif
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define G__ANSIHEADER
#define G__DICTIONARY
#include "G__ci.h"
extern "C" {
extern void G__cpp_setup_tagtablemydico();
extern void G__cpp_setup_inheritancemydico();
extern void G__cpp_setup_typetablemydico();
extern void G__cpp_setup_memvarmydico();
extern void G__cpp_setup_globalmydico();
extern void G__cpp_setup_memfuncmydico();
extern void G__cpp_setup_funcmydico();
extern void G__set_cpp_environmentmydico();
}


#include "TROOT.h"
#include "TMemberInspector.h"
