/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file main/G__main.c
 ************************************************************************
 * Description:
 *  C version main function
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>

extern void G__setothermain();

int main(argc,argv)
int argc;
char *argv[];
{
  G__setothermain(0);
  return(G__main(argc,argv));
}
