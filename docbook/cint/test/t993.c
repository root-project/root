/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#if defined(interp) && defined(makecint)
#pragma include "test.dll"
#else
#include "t993.h"
#endif

#ifndef __CINT__
#include <stdio.h>
#endif

void some_script_func( PSOME_STRUCT pStruc)
{
/* prepare to access the union data */
   PSOME_STRUCT2  pStruc2 = &(pStruc->u.s2);
   PSOME_STRUCT2* ppStruc2 = &pStruc2;
   printf("pStruc2->a     s2::a=%d\n",pStruc2->a);
   printf("(*ppStruc2)->a s2::a=%d\n",(*ppStruc2)->a);

   printf("pStruc->u.s2.a s2::a=%d\n",pStruc->u.s2.a);
   printf("pStruc->u.s3.b s3::b=%d\n",pStruc->u.s3.b);
   printf("pStruc->u.s3.c s3::c=%d\n",pStruc->u.s3.c);
}

int main() {
  SOME_STRUCT s;
  s.u.s3.b=12;
  s.u.s3.c=34;
  some_script_func(&s);
  return 0;
}
