/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include "Complex.h"

main()
{
  struct Complex a,b,c,d;

  ComplexInit();

  ComplexSet(&a,2,-2);
  ComplexDisplay(a);

  b=ComplexExp(a);
  ComplexDisplay(b);

  c=ComplexAdd(a,b);
  ComplexDisplay(c);

  d=ComplexMultiply(a,b);
  ComplexDisplay(d);

  printf("%g %g %g %g\n"
	 ,ComplexImag(j) ,ComplexAbs(b) ,ComplexReal(c) ,ComplexImag(d));
}

