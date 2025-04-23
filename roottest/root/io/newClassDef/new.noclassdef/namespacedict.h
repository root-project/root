/********************************************************************
* namespacedict.h
********************************************************************/
#ifdef __CINT__
#error namespacedict.h/C is only for compilation. Abort cint.
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
extern void G__cpp_setup_tagtablenamespacedict();
extern void G__cpp_setup_inheritancenamespacedict();
extern void G__cpp_setup_typetablenamespacedict();
extern void G__cpp_setup_memvarnamespacedict();
extern void G__cpp_setup_globalnamespacedict();
extern void G__cpp_setup_memfuncnamespacedict();
extern void G__cpp_setup_funcnamespacedict();
extern void G__set_cpp_environmentnamespacedict();
}


#include "TROOT.h"
#include "TMemberInspector.h"
#include "namespace.h"
#include <algorithm>

#ifndef G__MEMFUNCBODY
#endif

extern G__linked_taginfo G__namespacedictLN_TClass;
extern G__linked_taginfo G__namespacedictLN_TObject;
extern G__linked_taginfo G__namespacedictLN_MyClass0;
extern G__linked_taginfo G__namespacedictLN_MySpace;
extern G__linked_taginfo G__namespacedictLN_MySpacecLcLA;
extern G__linked_taginfo G__namespacedictLN_MySpacecLcLMyClass;
extern G__linked_taginfo G__namespacedictLN_vectorlEMySpacecLcLAcO__malloc_alloc_templatelE0gRsPgR;
extern G__linked_taginfo G__namespacedictLN_vectorlEMySpacecLcLAcO__malloc_alloc_templatelE0gRsPgRcLcLreverse_iterator;
extern G__linked_taginfo G__namespacedictLN_random_access_iteratorlEMySpacecLcLAcOlonggR;

/* STUB derived class for protected member access */
