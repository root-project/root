/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/********************************************************
* stdstrct.c
********************************************************/
#include "stdstrct.h"

void G__c_reset_tagtableG__stdstrct();
void G__set_c_environmentG__stdstrct() {
  G__add_compiledheader("stdstr.h");
  G__c_reset_tagtableG__stdstrct();
}
int G__c_dllrevG__stdstrct() { return(30051515); }

/* Setting up global function */

/*********************************************************
* Global function Stub
*********************************************************/

/*********************************************************
* typedef information setup/
*********************************************************/
void G__c_setup_typetableG__stdstrct() {

   /* Setting up typedef entry */
   G__search_typename2("div_t",117,G__get_linked_tagnum(&G__G__stdstrctLN_dAdiv_t),0,-1);
   G__setnewtype(-2,NULL,0);
   G__search_typename2("ldiv_t",117,G__get_linked_tagnum(&G__G__stdstrctLN_dAldiv_t),0,-1);
   G__setnewtype(-2,NULL,0);
}

/*********************************************************
* Data Member information setup/
*********************************************************/

   /* Setting up class,struct,union tag member variable */

   /* struct lconv */
static void G__setup_memvarlconv() {
   G__tag_memvar_setup(G__get_linked_tagnum(&G__G__stdstrctLN_lconv));
   { struct lconv *p; p=(struct lconv*)0x1000; if (p) { }
   G__memvar_setup((void*)((long)(&p->currency_symbol)-(long)(p)),67,0,0,-1,-1,-1,1,"currency_symbol=",0,(char*)NULL);
   G__memvar_setup((void*)((long)(&p->decimal_point)-(long)(p)),67,0,0,-1,-1,-1,1,"decimal_point=",0,(char*)NULL);
   G__memvar_setup((void*)((long)(&p->frac_digits)-(long)(p)),99,0,0,-1,-1,-1,1,"frac_digits=",0,(char*)NULL);
   G__memvar_setup((void*)((long)(&p->grouping)-(long)(p)),67,0,0,-1,-1,-1,1,"grouping=",0,(char*)NULL);
   G__memvar_setup((void*)((long)(&p->int_curr_symbol)-(long)(p)),67,0,0,-1,-1,-1,1,"int_curr_symbol=",0,(char*)NULL);
   G__memvar_setup((void*)((long)(&p->mon_decimal_point)-(long)(p)),67,0,0,-1,-1,-1,1,"mon_decimal_point=",0,(char*)NULL);
   G__memvar_setup((void*)((long)(&p->mon_grouping)-(long)(p)),67,0,0,-1,-1,-1,1,"mon_grouping=",0,(char*)NULL);
   G__memvar_setup((void*)((long)(&p->n_cs_precedes)-(long)(p)),99,0,0,-1,-1,-1,1,"n_cs_precedes=",0,(char*)NULL);
   G__memvar_setup((void*)((long)(&p->n_sep_by_space)-(long)(p)),99,0,0,-1,-1,-1,1,"n_sep_by_space=",0,(char*)NULL);
   G__memvar_setup((void*)((long)(&p->n_sign_posn)-(long)(p)),99,0,0,-1,-1,-1,1,"n_sign_posn=",0,(char*)NULL);
   G__memvar_setup((void*)((long)(&p->negative_sign)-(long)(p)),67,0,0,-1,-1,-1,1,"negative_sign=",0,(char*)NULL);
   G__memvar_setup((void*)((long)(&p->p_cs_precedes)-(long)(p)),99,0,0,-1,-1,-1,1,"p_cs_precedes=",0,(char*)NULL);
   G__memvar_setup((void*)((long)(&p->p_sep_by_space)-(long)(p)),99,0,0,-1,-1,-1,1,"p_sep_by_space=",0,(char*)NULL);
   G__memvar_setup((void*)((long)(&p->p_sign_posn)-(long)(p)),99,0,0,-1,-1,-1,1,"p_sign_posn=",0,(char*)NULL);
   G__memvar_setup((void*)((long)(&p->positive_sign)-(long)(p)),67,0,0,-1,-1,-1,1,"positive_sign=",0,(char*)NULL);
   G__memvar_setup((void*)((long)(&p->thousands_sep)-(long)(p)),67,0,0,-1,-1,-1,1,"thousands_sep=",0,(char*)NULL);
   }
   G__tag_memvar_reset();
}


   /* struct tm */
static void G__setup_memvartm() {
   G__tag_memvar_setup(G__get_linked_tagnum(&G__G__stdstrctLN_tm));
   { struct tm *p; p=(struct tm*)0x1000; if (p) { }
   G__memvar_setup((void*)((long)(&p->tm_sec)-(long)(p)),105,0,0,-1,-1,-1,1,"tm_sec=",0,(char*)NULL);
   G__memvar_setup((void*)((long)(&p->tm_min)-(long)(p)),105,0,0,-1,-1,-1,1,"tm_min=",0,(char*)NULL);
   G__memvar_setup((void*)((long)(&p->tm_hour)-(long)(p)),105,0,0,-1,-1,-1,1,"tm_hour=",0,(char*)NULL);
   G__memvar_setup((void*)((long)(&p->tm_mday)-(long)(p)),105,0,0,-1,-1,-1,1,"tm_mday=",0,(char*)NULL);
   G__memvar_setup((void*)((long)(&p->tm_mon)-(long)(p)),105,0,0,-1,-1,-1,1,"tm_mon=",0,(char*)NULL);
   G__memvar_setup((void*)((long)(&p->tm_year)-(long)(p)),105,0,0,-1,-1,-1,1,"tm_year=",0,(char*)NULL);
   G__memvar_setup((void*)((long)(&p->tm_wday)-(long)(p)),105,0,0,-1,-1,-1,1,"tm_wday=",0,(char*)NULL);
   G__memvar_setup((void*)((long)(&p->tm_yday)-(long)(p)),105,0,0,-1,-1,-1,1,"tm_yday=",0,(char*)NULL);
   G__memvar_setup((void*)((long)(&p->tm_isdst)-(long)(p)),105,0,0,-1,-1,-1,1,"tm_isdst=",0,(char*)NULL);
   }
   G__tag_memvar_reset();
}


   /* div_t */
static void G__setup_memvardAdiv_t() {
   G__tag_memvar_setup(G__get_linked_tagnum(&G__G__stdstrctLN_dAdiv_t));
   { div_t *p; p=(div_t*)0x1000; if (p) { }
   G__memvar_setup((void*)((long)(&p->quot)-(long)(p)),105,0,0,-1,-1,-1,1,"quot=",0,(char*)NULL);
   G__memvar_setup((void*)((long)(&p->rem)-(long)(p)),105,0,0,-1,-1,-1,1,"rem=",0,(char*)NULL);
   }
   G__tag_memvar_reset();
}


   /* ldiv_t */
static void G__setup_memvardAldiv_t() {
   G__tag_memvar_setup(G__get_linked_tagnum(&G__G__stdstrctLN_dAldiv_t));
   { ldiv_t *p; p=(ldiv_t*)0x1000; if (p) { }
   G__memvar_setup((void*)((long)(&p->quot)-(long)(p)),108,0,0,-1,-1,-1,1,"quot=",0,(char*)NULL);
   G__memvar_setup((void*)((long)(&p->rem)-(long)(p)),108,0,0,-1,-1,-1,1,"rem=",0,(char*)NULL);
   }
   G__tag_memvar_reset();
}

void G__c_setup_memvarG__stdstrct() {
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

   G__memvar_setup((void*)G__PVOID,112,0,0,-1,-1,-1,1,"G__STDSTRUCT=0",1,(char*)NULL);

   G__resetglobalenv();
}
void G__c_setup_globalG__stdstrct() {
  G__cpp_setup_global0();
}

/*********************************************************
* Global function information setup for each class
*********************************************************/
static void G__cpp_setup_func0() {
   G__lastifuncposition();


   G__resetifuncposition();
}

void G__c_setup_funcG__stdstrct() {
  G__cpp_setup_func0();
}

/*********************************************************
* Class,struct,union,enum tag information setup
*********************************************************/
/* Setup class/struct taginfo */
G__linked_taginfo G__G__stdstrctLN_lconv = { "lconv" , 115 , -1 };
G__linked_taginfo G__G__stdstrctLN_tm = { "tm" , 115 , -1 };
G__linked_taginfo G__G__stdstrctLN_dAdiv_t = { "$div_t" , 115 , -1 };
G__linked_taginfo G__G__stdstrctLN_dAldiv_t = { "$ldiv_t" , 115 , -1 };

/* Reset class/struct taginfo */
void G__c_reset_tagtableG__stdstrct() {
  G__G__stdstrctLN_lconv.tagnum = -1 ;
  G__G__stdstrctLN_tm.tagnum = -1 ;
  G__G__stdstrctLN_dAdiv_t.tagnum = -1 ;
  G__G__stdstrctLN_dAldiv_t.tagnum = -1 ;
}


void G__c_setup_tagtableG__stdstrct() {

   /* Setting up class,struct,union tag entry */
   G__tagtable_setup(G__get_linked_tagnum(&G__G__stdstrctLN_lconv),sizeof(struct lconv),-2,0,(char*)NULL,G__setup_memvarlconv,NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__stdstrctLN_tm),sizeof(struct tm),-2,0,(char*)NULL,G__setup_memvartm,NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__stdstrctLN_dAdiv_t),sizeof(div_t),-2,0,(char*)NULL,G__setup_memvardAdiv_t,NULL);
   G__tagtable_setup(G__get_linked_tagnum(&G__G__stdstrctLN_dAldiv_t),sizeof(ldiv_t),-2,0,(char*)NULL,G__setup_memvardAldiv_t,NULL);
}

extern "C" void G__c_setupG__stdstrct() {
  G__check_setup_version(30051515,"G__c_setupG__stdstrct()");
  G__set_c_environmentG__stdstrct();
  G__c_setup_tagtableG__stdstrct();

  G__c_setup_typetableG__stdstrct();

  G__c_setup_memvarG__stdstrct();

  G__c_setup_globalG__stdstrct();
  G__c_setup_funcG__stdstrct();
  return;
}
