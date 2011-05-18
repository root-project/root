// @(#)root/main:$Id$
// Author: Masaharu Goto   16/05/2000
/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file main/G__cppmain.C
 ************************************************************************
 * Description:
 *  C++ version main function
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto (MXJ02154@niftyserve.or.jp)
 *
 * Permission to use, copy, modify and distribute this software and its
 * documentation for non-commercial purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  The author makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 ************************************************************************/
#include <stdio.h>

extern "C" {
extern void G__setothermain(int othermain);
extern int G__main(int argc,char **argv);
}

int main(int argc,char **argv)
{
   G__setothermain(0);
   return(G__main(argc,argv));
}
