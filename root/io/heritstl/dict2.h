/********************************************************************
* dict2.h
********************************************************************/
#ifdef __CINT__
#error dict2.h/C is only for compilation. Abort cint.
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
extern void G__cpp_setup_tagtabledict2();
extern void G__cpp_setup_inheritancedict2();
extern void G__cpp_setup_typetabledict2();
extern void G__cpp_setup_memvardict2();
extern void G__cpp_setup_globaldict2();
extern void G__cpp_setup_memfuncdict2();
extern void G__cpp_setup_funcdict2();
extern void G__set_cpp_environmentdict2();
}


#include "TROOT.h"
#include "TMemberInspector.h"
#include "classes.C"
#include <algorithm>
namespace std { }
using namespace std;

#ifndef G__MEMFUNCBODY
#endif

extern G__linked_taginfo G__dict2LN_TClass;
extern G__linked_taginfo G__dict2LN_TObject;
extern G__linked_taginfo G__dict2LN_TNamed;
extern G__linked_taginfo G__dict2LN_random_access_iterator_tag;
extern G__linked_taginfo G__dict2LN_PlexItem;
extern G__linked_taginfo G__dict2LN_PlexSTL;
extern G__linked_taginfo G__dict2LN_vectorlEPlexItemcOallocatorlEPlexItemgRsPgR;
extern G__linked_taginfo G__dict2LN___normal_iteratorlEPlexItemmUcOvectorlEPlexItemcOallocatorlEPlexItemgRsPgRsPgR;
extern G__linked_taginfo G__dict2LN_iterator_traitslEPlexItemmUgR;
extern G__linked_taginfo G__dict2LN_iteratorlErandom_access_iterator_tagcOPlexItemcOlongcOPlexItemmUcOPlexItemaNgR;
extern G__linked_taginfo G__dict2LN_vectorlEPlexItemcOallocatorlEPlexItemgRsPgRcLcLreverse_iterator;
extern G__linked_taginfo G__dict2LN_random_access_iteratorlEPlexItemcOlonggR;
extern G__linked_taginfo G__dict2LN_Object;

/* STUB derived class for protected member access */
typedef vector<PlexItem,allocator<PlexItem> > G__vectorlEPlexItemcOallocatorlEPlexItemgRsPgR;
