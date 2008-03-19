/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/********************************************************************
* stdstrct.h
********************************************************************/
#ifdef __CINT__
#error stdstrct.h/C is only for compilation. Abort cint.
#endif
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define G__ANSIHEADER
#define G__DICTIONARY
#include "G__ci.h"
extern void G__c_setup_tagtableG__stdstrct();
extern void G__c_setup_typetableG__stdstrct();
extern void G__c_setup_memvarG__stdstrct();
extern void G__c_setup_globalG__stdstrct();
extern void G__c_setup_funcG__stdstrct();
extern void G__set_c_environmentG__stdstrct();


#include "stdstr.h"
extern G__linked_taginfo G__G__stdstrctLN_lconv;
extern G__linked_taginfo G__G__stdstrctLN_tm;
extern G__linked_taginfo G__G__stdstrctLN_dAdiv_t;
extern G__linked_taginfo G__G__stdstrctLN_dAldiv_t;

/* STUB derived class for protected member access */
