/********************************************************************
* dict.h
********************************************************************/
#ifdef __CINT__
#error dict.h/C is only for compilation. Abort cint.
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
extern void G__cpp_setup_tagtabledict();
extern void G__cpp_setup_inheritancedict();
extern void G__cpp_setup_typetabledict();
extern void G__cpp_setup_memvardict();
extern void G__cpp_setup_globaldict();
extern void G__cpp_setup_memfuncdict();
extern void G__cpp_setup_funcdict();
extern void G__set_cpp_environmentdict();
}


#include "TROOT.h"
#include "TMemberInspector.h"
#include "test.h"

#ifndef G__MEMFUNCBODY
#endif

extern G__linked_taginfo G__dictLN_A;

/* STUB derived class for protected member access */
