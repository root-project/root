/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/****************************************************************
* setjmp.h
*****************************************************************/
#ifndef G__SETJMP_H
#define G__SETJMP_H

// #define G__WARNING_AT_DECLARATION

class jmp_buf {
 public:
  jmp_buf(void) {
#ifdef G__WARNING_AT_DECLARATION
    fprintf(stderr,"Limitation: jmp_buf not supported\n");
#endif
  }
};

typedef jmp_buf sigjmp_buf;

setjmp(jmp_buf& buf)
{
  fprintf(stderr,"Limitation: setjmp() not supported\n");
  return(0);
}



void longjmp(jmp_buf environment,int rval)
{
  fprintf(stderr,"Limitation: longjmp() not supported\n");
}

#endif
