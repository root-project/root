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
namespace std { }
using namespace std;

#ifndef G__MEMFUNCBODY
#endif

extern G__linked_taginfo G__EventDictLN_TClass;
extern G__linked_taginfo G__EventDictLN_TBuffer;
extern G__linked_taginfo G__EventDictLN_TMemberInspector;
extern G__linked_taginfo G__EventDictLN_TObject;
extern G__linked_taginfo G__EventDictLN_string;
extern G__linked_taginfo G__EventDictLN_TObjArray;
extern G__linked_taginfo G__EventDictLN_TString;
extern G__linked_taginfo G__EventDictLN_TDatime;
extern G__linked_taginfo G__EventDictLN_TDirectory;
extern G__linked_taginfo G__EventDictLN_TClonesArray;
extern G__linked_taginfo G__EventDictLN_TAxis;
extern G__linked_taginfo G__EventDictLN_TArrayI;
extern G__linked_taginfo G__EventDictLN_TArrayF;
extern G__linked_taginfo G__EventDictLN_TH1F;
extern G__linked_taginfo G__EventDictLN_TRef;
extern G__linked_taginfo G__EventDictLN_TLorentzVector;
extern G__linked_taginfo G__EventDictLN_UShortVector;
extern G__linked_taginfo G__EventDictLN_vectorlEunsignedsPshortcOallocatorlEunsignedsPshortgRsPgR;
extern G__linked_taginfo G__EventDictLN_EventHeader;
extern G__linked_taginfo G__EventDictLN_Event;
extern G__linked_taginfo G__EventDictLN_EventcLcLdA;
extern G__linked_taginfo G__EventDictLN_vectorlEintcOallocatorlEintgRsPgR;
extern G__linked_taginfo G__EventDictLN_vectorlEshortcOallocatorlEshortgRsPgR;
extern G__linked_taginfo G__EventDictLN_vectorlEdoublecOallocatorlEdoublegRsPgR;
extern G__linked_taginfo G__EventDictLN_vectorlETLinecOallocatorlETLinegRsPgR;
extern G__linked_taginfo G__EventDictLN_vectorlETObjectcOallocatorlETObjectgRsPgR;
extern G__linked_taginfo G__EventDictLN_vectorlETNamedcOallocatorlETNamedgRsPgR;
extern G__linked_taginfo G__EventDictLN_vectorlEstringcOallocatorlEstringgRsPgR;
extern G__linked_taginfo G__EventDictLN_dequelETAttLinecOallocatorlETAttLinegRsPgR;
extern G__linked_taginfo G__EventDictLN_listlEconstsPTObjectmUcOallocatorlEconstsPTObjectmUgRsPgR;
extern G__linked_taginfo G__EventDictLN_listlEstringcOallocatorlEstringgRsPgR;
extern G__linked_taginfo G__EventDictLN_listlEstringmUcOallocatorlEstringmUgRsPgR;
extern G__linked_taginfo G__EventDictLN_maplETNamedmUcOintcOlesslETNamedmUgRcOallocatorlEpairlEconstsPTNamedmUcOintgRsPgRsPgR;
extern G__linked_taginfo G__EventDictLN_template1lEintgR;
extern G__linked_taginfo G__EventDictLN_template2lEtemplate1lEintgRsPgR;
extern G__linked_taginfo G__EventDictLN_maplETStringcOTListmUcOlesslETStringgRcOallocatorlEpairlEconstsPTStringcOTListmUgRsPgRsPgR;
extern G__linked_taginfo G__EventDictLN_maplETStringcOTStringmUcOlesslETStringgRcOallocatorlEpairlEconstsPTStringcOTStringmUgRsPgRsPgR;
extern G__linked_taginfo G__EventDictLN_maplEEventHeadercOTStringmUcOlesslEEventHeadergRcOallocatorlEpairlEconstsPEventHeadercOTStringmUgRsPgRsPgR;
extern G__linked_taginfo G__EventDictLN_maplEEventHeadercOTStringcOlesslEEventHeadergRcOallocatorlEpairlEconstsPEventHeadercOTStringgRsPgRsPgR;
extern G__linked_taginfo G__EventDictLN_maplEEventHeadercOstringcOlesslEEventHeadergRcOallocatorlEpairlEconstsPEventHeadercOstringgRsPgRsPgR;
extern G__linked_taginfo G__EventDictLN_maplETAxismUcOintcOlesslETAxismUgRcOallocatorlEpairlEconstsPTAxismUcOintgRsPgRsPgR;
extern G__linked_taginfo G__EventDictLN_setlETAxismUcOlesslETAxismUgRcOallocatorlETAxismUgRsPgR;
extern G__linked_taginfo G__EventDictLN_multimaplETNamedmUcOintcOlesslETNamedmUgRcOallocatorlEintgRsPgR;
extern G__linked_taginfo G__EventDictLN_multisetlETAxismUcOlesslETAxismUgRcOallocatorlETAxismUgRsPgR;
extern G__linked_taginfo G__EventDictLN_vectorlETAxismUcOallocatorlETAxismUgRsPgR;
extern G__linked_taginfo G__EventDictLN_vectorlEvectorlETAxismUcOallocatorlETAxismUgRsPgRcOallocatorlEvectorlETAxismUcOallocatorlETAxismUgRsPgRsPgRsPgR;
extern G__linked_taginfo G__EventDictLN_maplEstringcOvectorlEintcOallocatorlEintgRsPgRcOlesslEstringgRcOallocatorlEpairlEconstsPstringcOvectorlEintcOallocatorlEintgRsPgRsPgRsPgRsPgR;
extern G__linked_taginfo G__EventDictLN_dequelEpairlEfloatcOfloatgRcOallocatorlEpairlEfloatcOfloatgRsPgRsPgR;
extern G__linked_taginfo G__EventDictLN_Track;
extern G__linked_taginfo G__EventDictLN_BigTrack;
extern G__linked_taginfo G__EventDictLN_HistogramManager;

/* STUB derived class for protected member access */
typedef template1<int> G__template1lEintgR;
typedef template2<template1<int> > G__template2lEtemplate1lEintgRsPgR;
