/********************************************************************
* dummy.h
********************************************************************/
#ifdef __CINT__
#error dummy.h/C is only for compilation. Abort cint.
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
extern void G__cpp_setup_tagtabledummy();
extern void G__cpp_setup_inheritancedummy();
extern void G__cpp_setup_typetabledummy();
extern void G__cpp_setup_memvardummy();
extern void G__cpp_setup_globaldummy();
extern void G__cpp_setup_memfuncdummy();
extern void G__cpp_setup_funcdummy();
extern void G__set_cpp_environmentdummy();
}


#include "TROOT.h"
#include "TMemberInspector.h"
#include "Simple.cxx"

#ifndef G__MEMFUNCBODY
#endif

extern G__linked_taginfo G__dummyLN_TClass;
extern G__linked_taginfo G__dummyLN_TObject;
extern G__linked_taginfo G__dummyLN__x3d_data_;
extern G__linked_taginfo G__dummyLN__x3d_sizeof_;
extern G__linked_taginfo G__dummyLN_TShape;
extern G__linked_taginfo G__dummyLN_Simple;

/* STUB derived class for protected member access */
