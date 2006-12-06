/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file dmystrm.c
 ************************************************************************
 * Description:
 *  iostream and ERTTI API dummy function for C compiler only intallation
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <stdio.h> 
#include "common.h"

extern "C" {

extern FILE *G__serr;

void G__cpp_setupG__stream() {
  /* dummy */
}

void G__cpp_setupG__API() {
  /* dummy */
  if(G__dispmsg>=G__DISPWARN) {
    G__fprinterr(G__serr,
         "Warning: ERTTI API not available. Install CINT with C++ compiler\n");
  }
}

#ifndef G__OLDIMPLEMENTAITON472
/***********************************************************************
* set linkage of precompiled library function
***********************************************************************/
int G__SetGlobalcomp(char *funcname,char *param, int globalcomp)
{
  int i;
  struct G__ifunc_table *ifunc;

  ifunc = &G__ifunc;
  while(ifunc) {
    for(i=0;i<ifunc->allifunc;i++) {
      if(strcmp(funcname,ifunc->funcname[i])==0) {
        ifunc->globalcomp[i] = globalcomp;
      }
    }
    ifunc = ifunc->next;
  }
  return(0);
}
#endif

int G__ForceBytecodecompilation(char *funcname,char *param)
{
  return(0);
}

void G__redirectcout(char *filename)
{
}
void G__unredirectcout() {
}
void G__redirectcerr(char *filename)
{
}
void G__unredirectcerr() {
}
void G__redirectcin(char *filename)
{
}
void G__unredirectcin() {
}

}
