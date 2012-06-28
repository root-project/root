/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/****************************************************************
* stdarg.h
*****************************************************************/
#ifndef G__STDARG_H
#define G__STDARG_H

struct va_list {
  void* libp;
  int    ip;
} ;

#ifdef G__OLDIMPLEMENTATION1472
/* not needed anymore */
void va_start(va_list listptr,int dummy) {
  fprintf(stderr,"Limitation: va_start(),va_arg(),va_end() not supported\n");
}
#endif

#endif
