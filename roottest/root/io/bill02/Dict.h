/********************************************************************
* Dict.h
********************************************************************/
#ifdef __CINT__
#error Dict.h/C is only for compilation. Abort cint.
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
extern void G__cpp_setup_tagtableDict();
extern void G__cpp_setup_inheritanceDict();
extern void G__cpp_setup_typetableDict();
extern void G__cpp_setup_memvarDict();
extern void G__cpp_setup_globalDict();
extern void G__cpp_setup_memfuncDict();
extern void G__cpp_setup_funcDict();
extern void G__set_cpp_environmentDict();
}


#include "TROOT.h"
#include "TMemberInspector.h"
#include "MyObject.h"
#include <algorithm>
namespace std { }
using namespace std;

#ifndef G__MEMFUNCBODY
#endif

extern G__linked_taginfo G__DictLN_TClass;
extern G__linked_taginfo G__DictLN_TObject;
extern G__linked_taginfo G__DictLN_BaseMyObject;
extern G__linked_taginfo G__DictLN_MyObject;

/* STUB derived class for protected member access */
