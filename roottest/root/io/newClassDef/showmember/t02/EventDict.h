/********************************************************************
* EventDict.h
********************************************************************/
#ifdef __CINT__
#error EventDict.h/C is only for compilation. Abort cint.
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
extern void G__cpp_setup_tagtableEventDict();
extern void G__cpp_setup_inheritanceEventDict();
extern void G__cpp_setup_typetableEventDict();
extern void G__cpp_setup_memvarEventDict();
extern void G__cpp_setup_globalEventDict();
extern void G__cpp_setup_memfuncEventDict();
extern void G__cpp_setup_funcEventDict();
extern void G__set_cpp_environmentEventDict();
}


#include "TROOT.h"
#include "TMemberInspector.h"
#include "Event.h"
#include <algorithm>

#ifndef G__MEMFUNCBODY
#endif

extern G__linked_taginfo G__EventDictLN_type_info;
extern G__linked_taginfo G__EventDictLN_TClass;
extern G__linked_taginfo G__EventDictLN_TObject;
extern G__linked_taginfo G__EventDictLN___malloc_alloc_templatelE0gR;
extern G__linked_taginfo G__EventDictLN_lesslEconstsPtype_infomUgR;
extern G__linked_taginfo G__EventDictLN_maplEconstsPtype_infomUcOTClassmUcOlesslEconstsPtype_infomUgRcO__malloc_alloc_templatelE0gRsPgR;
extern G__linked_taginfo G__EventDictLN_pairlEconstsPtype_infomUcOTClassmUgR;
extern G__linked_taginfo G__EventDictLN_maplEconstsPtype_infomUcOTClassmUcOlesslEconstsPtype_infomUgRcO__malloc_alloc_templatelE0gRsPgRcLcLiterator;
extern G__linked_taginfo G__EventDictLN_bidirectional_iteratorlETClassmUcOlonggR;
extern G__linked_taginfo G__EventDictLN_maplEconstsPtype_infomUcOTClassmUcOlesslEconstsPtype_infomUgRcO__malloc_alloc_templatelE0gRsPgRcLcLreverse_iterator;
extern G__linked_taginfo G__EventDictLN_TClonesArray;
extern G__linked_taginfo G__EventDictLN_TRefArray;
extern G__linked_taginfo G__EventDictLN_TRef;
extern G__linked_taginfo G__EventDictLN_TH1;
extern G__linked_taginfo G__EventDictLN_TH1F;
extern G__linked_taginfo G__EventDictLN_Track;
extern G__linked_taginfo G__EventDictLN_EventHeader;
extern G__linked_taginfo G__EventDictLN_Event;
extern G__linked_taginfo G__EventDictLN_HistogramManager;

/* STUB derived class for protected member access */
