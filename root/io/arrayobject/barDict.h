/********************************************************************
* barDict.h
********************************************************************/
#ifdef __CINT__
#error barDict.h/C is only for compilation. Abort cint.
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
extern void G__cpp_setup_tagtablebarDict();
extern void G__cpp_setup_inheritancebarDict();
extern void G__cpp_setup_typetablebarDict();
extern void G__cpp_setup_memvarbarDict();
extern void G__cpp_setup_globalbarDict();
extern void G__cpp_setup_memfuncbarDict();
extern void G__cpp_setup_funcbarDict();
extern void G__set_cpp_environmentbarDict();
}


#include "TROOT.h"
#include "TMemberInspector.h"
#include "bar.h"

#ifndef G__MEMFUNCBODY
#endif

extern G__linked_taginfo G__barDictLN_TClass;
extern G__linked_taginfo G__barDictLN_TObject;
extern G__linked_taginfo G__barDictLN_foobj;
extern G__linked_taginfo G__barDictLN_foo;
extern G__linked_taginfo G__barDictLN_bar;

/* STUB derived class for protected member access */
