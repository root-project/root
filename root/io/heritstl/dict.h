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
#include "classes.C"
#include <algorithm>
namespace std { }
using namespace std;

#ifndef G__MEMFUNCBODY
#endif

extern G__linked_taginfo G__dictLN_TClass;
extern G__linked_taginfo G__dictLN_TObject;
extern G__linked_taginfo G__dictLN_TNamed;
extern G__linked_taginfo G__dictLN_random_access_iterator_tag;
extern G__linked_taginfo G__dictLN_PlexItem;
extern G__linked_taginfo G__dictLN_PlexSTL;
extern G__linked_taginfo G__dictLN_vectorlEPlexItemcOallocatorlEPlexItemgRsPgR;
extern G__linked_taginfo G__dictLN___normal_iteratorlEPlexItemmUcOvectorlEPlexItemcOallocatorlEPlexItemgRsPgRsPgR;
extern G__linked_taginfo G__dictLN_iterator_traitslEPlexItemmUgR;
extern G__linked_taginfo G__dictLN_iteratorlErandom_access_iterator_tagcOPlexItemcOlongcOPlexItemmUcOPlexItemaNgR;
extern G__linked_taginfo G__dictLN_vectorlEPlexItemcOallocatorlEPlexItemgRsPgRcLcLreverse_iterator;
extern G__linked_taginfo G__dictLN_random_access_iteratorlEPlexItemcOlonggR;
extern G__linked_taginfo G__dictLN_Object;

/* STUB derived class for protected member access */
typedef vector<PlexItem,allocator<PlexItem> > G__vectorlEPlexItemcOallocatorlEPlexItemgRsPgR;
