/********************************************************************
* nstemplatedict.h
********************************************************************/
#ifdef __CINT__
#error nstemplatedict.h/C is only for compilation. Abort cint.
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
extern void G__cpp_setup_tagtablenstemplatedict();
extern void G__cpp_setup_inheritancenstemplatedict();
extern void G__cpp_setup_typetablenstemplatedict();
extern void G__cpp_setup_memvarnstemplatedict();
extern void G__cpp_setup_globalnstemplatedict();
extern void G__cpp_setup_memfuncnstemplatedict();
extern void G__cpp_setup_funcnstemplatedict();
extern void G__set_cpp_environmentnstemplatedict();
}


#include "TROOT.h"
#include "TMemberInspector.h"
#include "./Rtypes.h"
#include "nstemplate.h"
#include <algorithm>

#ifndef G__MEMFUNCBODY
#endif

extern G__linked_taginfo G__nstemplatedictLN_TClass;
extern G__linked_taginfo G__nstemplatedictLN_TObject;
extern G__linked_taginfo G__nstemplatedictLN_MySpace;
extern G__linked_taginfo G__nstemplatedictLN_vectorlEintcO__malloc_alloc_templatelE0gRsPgR;
extern G__linked_taginfo G__nstemplatedictLN_vectorlEintcO__malloc_alloc_templatelE0gRsPgRcLcLreverse_iterator;
extern G__linked_taginfo G__nstemplatedictLN_random_access_iteratorlEintcOlonggR;
extern G__linked_taginfo G__nstemplatedictLN_MySpacecLcLMyTemplatelEconstsPintmUgR;

/* STUB derived class for protected member access */
typedef MySpace::MyTemplate<const int*> G__MySpacecLcLMyTemplatelEconstsPintmUgR;
