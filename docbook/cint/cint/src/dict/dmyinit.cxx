/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/* dummy init routine for C++ defined symbol initialization */

#include <stdio.h>

extern "C" {

   void G__initcxx() 
   {
   }

   void G__init_replacesymbol() {}
   void G__add_replacesymbol(const char* s1,const char* s2) {}
   const char* G__replacesymbol(const char* s) { return(s); }
   int G__display_replacesymbol(FILE *fout,const char* name) { return(0); }

}