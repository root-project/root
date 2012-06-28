/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

/* include CINT header file */
#include "G__ci.h"

/* YOU MUST PRECOMPILE FOLLOWING FUNCTIONS */

void display(void)
{
  /* call interpreted function from compiled code */
  G__calc("displayBody()"); /* see doc/ref.txt for G__calc API */
}

void key(unsigned char k, int x, int y)
{
  /* call interpreted function from compiled code */
  char buf[200];
  sprintf(buf,"keyBody('%c',%d,%d)",k,x,y);
  G__calc(buf); /* see doc/ref.txt for G__calc API */
}
