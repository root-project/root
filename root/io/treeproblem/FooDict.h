/********************************************************************
* FooDict.h
********************************************************************/
#ifdef __CINT__
#error FooDict.h/C is only for compilation. Abort cint.
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
extern void G__cpp_setup_tagtableFooDict();
extern void G__cpp_setup_inheritanceFooDict();
extern void G__cpp_setup_typetableFooDict();
extern void G__cpp_setup_memvarFooDict();
extern void G__cpp_setup_globalFooDict();
extern void G__cpp_setup_memfuncFooDict();
extern void G__cpp_setup_funcFooDict();
extern void G__set_cpp_environmentFooDict();
}


#include "TROOT.h"
#include "TMemberInspector.h"
#include "Foo.h"

#ifndef G__MEMFUNCBODY
#endif

extern G__linked_taginfo G__FooDictLN_TClass;
extern G__linked_taginfo G__FooDictLN_TObject;
extern G__linked_taginfo G__FooDictLN_Foo;

/* STUB derived class for protected member access */
