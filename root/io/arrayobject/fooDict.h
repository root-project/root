/********************************************************************
* fooDict.h
********************************************************************/
#ifdef __CINT__
#error fooDict.h/C is only for compilation. Abort cint.
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
extern void G__cpp_setup_tagtablefooDict();
extern void G__cpp_setup_inheritancefooDict();
extern void G__cpp_setup_typetablefooDict();
extern void G__cpp_setup_memvarfooDict();
extern void G__cpp_setup_globalfooDict();
extern void G__cpp_setup_memfuncfooDict();
extern void G__cpp_setup_funcfooDict();
extern void G__set_cpp_environmentfooDict();
}


#include "TROOT.h"
#include "TMemberInspector.h"
#include "foo.h"

#ifndef G__MEMFUNCBODY
#endif

extern G__linked_taginfo G__fooDictLN_TClass;
extern G__linked_taginfo G__fooDictLN_TObject;
extern G__linked_taginfo G__fooDictLN_foobj;
extern G__linked_taginfo G__fooDictLN_foo;

/* STUB derived class for protected member access */
