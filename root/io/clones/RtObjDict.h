/********************************************************************
* RtObjDict.h
********************************************************************/
#ifdef __CINT__
#error RtObjDict.h/C is only for compilation. Abort cint.
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
extern void G__cpp_setup_tagtableRtObjDict();
extern void G__cpp_setup_inheritanceRtObjDict();
extern void G__cpp_setup_typetableRtObjDict();
extern void G__cpp_setup_memvarRtObjDict();
extern void G__cpp_setup_globalRtObjDict();
extern void G__cpp_setup_memfuncRtObjDict();
extern void G__cpp_setup_funcRtObjDict();
extern void G__set_cpp_environmentRtObjDict();
}


#include "TROOT.h"
#include "TMemberInspector.h"
#include "RtObj.h"

#ifndef G__MEMFUNCBODY
#endif

extern G__linked_taginfo G__RtObjDictLN_TClass;
extern G__linked_taginfo G__RtObjDictLN_TObject;
extern G__linked_taginfo G__RtObjDictLN_TNamed;
extern G__linked_taginfo G__RtObjDictLN_RtObjlETNamedgR;

/* STUB derived class for protected member access */
typedef RtObj<TNamed> G__RtObjlETNamedgR;
