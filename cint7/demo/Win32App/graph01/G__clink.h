/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/********************************************************************
* G__clink.h
********************************************************************/
#ifdef __CINT__
#error G__clink.h/C is only for compilation. Abort cint.
#endif
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define G__ANSIHEADER
#define G__DICTIONARY
#include "G__ci.h"
extern void G__c_setup_tagtable();
extern void G__c_setup_typetable();
extern void G__c_setup_memvar();
extern void G__c_setup_global();
extern void G__c_setup_func();
extern void G__set_c_environment();


#include "CompiledLib.h"

/* STUB derived class for protected member access */
