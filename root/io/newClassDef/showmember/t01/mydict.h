/********************************************************************
* mydict.h
********************************************************************/
#ifdef __CINT__
#error mydict.h/C is only for compilation. Abort cint.
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
extern void G__cpp_setup_tagtablemydict();
extern void G__cpp_setup_inheritancemydict();
extern void G__cpp_setup_typetablemydict();
extern void G__cpp_setup_memvarmydict();
extern void G__cpp_setup_globalmydict();
extern void G__cpp_setup_memfuncmydict();
extern void G__cpp_setup_funcmydict();
extern void G__set_cpp_environmentmydict();
}


#include "TROOT.h"
#include "TMemberInspector.h"
#include "RootData.h"
#include "RootCaloHit.h"

#ifndef G__MEMFUNCBODY
#endif

extern G__linked_taginfo G__mydictLN_TClass;
extern G__linked_taginfo G__mydictLN_TObject;
extern G__linked_taginfo G__mydictLN_TNamed;
extern G__linked_taginfo G__mydictLN_ROOT;
extern G__linked_taginfo G__mydictLN_TClonesArray;
extern G__linked_taginfo G__mydictLN_RootData;
extern G__linked_taginfo G__mydictLN_crap;
extern G__linked_taginfo G__mydictLN_RootPCellID;
extern G__linked_taginfo G__mydictLN_RootPCfix;
extern G__linked_taginfo G__mydictLN_RootPCvirt;
extern G__linked_taginfo G__mydictLN_RootPCnodict;
extern G__linked_taginfo G__mydictLN_RootCaloHit;

/* STUB derived class for protected member access */
