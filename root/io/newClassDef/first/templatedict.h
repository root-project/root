/********************************************************************
* templatedict.h
********************************************************************/
#ifdef __CINT__
#error templatedict.h/C is only for compilation. Abort cint.
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
extern void G__cpp_setup_tagtabletemplatedict();
extern void G__cpp_setup_inheritancetemplatedict();
extern void G__cpp_setup_typetabletemplatedict();
extern void G__cpp_setup_memvartemplatedict();
extern void G__cpp_setup_globaltemplatedict();
extern void G__cpp_setup_memfunctemplatedict();
extern void G__cpp_setup_functemplatedict();
extern void G__set_cpp_environmenttemplatedict();
}


#include "TROOT.h"
#include "TMemberInspector.h"
#include "./Rtypes.h"
#include "template.h"
#include <algorithm>

#ifndef G__MEMFUNCBODY
#endif

extern G__linked_taginfo G__templatedictLN_TClass;
extern G__linked_taginfo G__templatedictLN_TObject;
extern G__linked_taginfo G__templatedictLN_MyTemplatelEconstsPdoublemUgR;
extern G__linked_taginfo G__templatedictLN_vectorlEintcO__malloc_alloc_templatelE0gRsPgR;
extern G__linked_taginfo G__templatedictLN_vectorlEintcO__malloc_alloc_templatelE0gRsPgRcLcLreverse_iterator;
extern G__linked_taginfo G__templatedictLN_random_access_iteratorlEintcOlonggR;
extern G__linked_taginfo G__templatedictLN_MyPairTemplatelEintcOdoublegR;
extern G__linked_taginfo G__templatedictLN_MyTemplatelEconstsPintmUgR;
extern G__linked_taginfo G__templatedictLN_MyPairTemplatelEintcOintgR;

/* STUB derived class for protected member access */
typedef MyTemplate<const double*> G__MyTemplatelEconstsPdoublemUgR;
typedef MyPairTemplate<int,double> G__MyPairTemplatelEintcOdoublegR;
typedef MyTemplate<const int*> G__MyTemplatelEconstsPintmUgR;
typedef MyPairTemplate<int,int> G__MyPairTemplatelEintcOintgR;
