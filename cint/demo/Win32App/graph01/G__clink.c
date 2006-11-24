/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/********************************************************
* G__clink.c
********************************************************/
#include "G__clink.h"
void G__c_reset_tagtable();
void G__set_c_environment() {
  G__add_compiledheader("CompiledLib.h");
  G__add_compiledheader("<windows.h");
  G__c_reset_tagtable();
}
int G__c_dllrev() { return(30051515); }

/* Setting up global function */
static int G___DrawRect4_5_14(result7,funcname,libp,hash)
G__value *result7;
char *funcname;
struct G__param *libp;
int hash;
 {
      G__setnull(result7);
      DrawRect4((HDC)G__int(libp->para[0]));
   return(1 || funcname || hash || result7 || libp) ;
}


/*********************************************************
* Global function Stub
*********************************************************/

/*********************************************************
* typedef information setup/
*********************************************************/
void G__c_setup_typetable() {

   /* Setting up typedef entry */
}

/*********************************************************
* Data Member information setup/
*********************************************************/

   /* Setting up class,struct,union tag member variable */
void G__c_setup_memvar() {
}
/***********************************************************
************************************************************
************************************************************
************************************************************
************************************************************
************************************************************
************************************************************
***********************************************************/

/*********************************************************
* Global variable information setup for each class
*********************************************************/
static void G__cpp_setup_global0() {

   /* Setting up global variables */
   G__resetplocal();

}

static void G__cpp_setup_global1() {
}

static void G__cpp_setup_global2() {
}

static void G__cpp_setup_global3() {

   G__resetglobalenv();
}
void G__c_setup_global() {
  G__cpp_setup_global0();
  G__cpp_setup_global1();
  G__cpp_setup_global2();
  G__cpp_setup_global3();
}

/*********************************************************
* Global function information setup for each class
*********************************************************/
static void G__cpp_setup_func0() {
   G__lastifuncposition();

}

static void G__cpp_setup_func1() {
   G__memfunc_setup("DrawRect4",848,G___DrawRect4_5_14,121,-1,-1,0,1,1,1,0,"U 'HDC__' 'HDC' 0 - hdc",(char*)NULL
#ifndef DrawRect4
,(void*)DrawRect4,0);
#else
,(void*)NULL,0);
#endif

   G__resetifuncposition();
}

void G__c_setup_func() {
  G__cpp_setup_func0();
  G__cpp_setup_func1();
}

/*********************************************************
* Class,struct,union,enum tag information setup
*********************************************************/
/* Setup class/struct taginfo */

/* Reset class/struct taginfo */
void G__c_reset_tagtable() {
}


void G__c_setup_tagtable() {

   /* Setting up class,struct,union tag entry */
}
void G__c_setup() {
  G__check_setup_version(30051515,"G__c_setup()");
  G__set_c_environment();
  G__c_setup_tagtable();

  G__c_setup_typetable();

  G__c_setup_memvar();

  G__c_setup_global();
  G__c_setup_func();
  return;
}
