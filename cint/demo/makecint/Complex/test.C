/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
//
// test.c : test for simple Complex class
//

#ifndef COMPLEX_H
#error "You must embed Complex.h/C to run this program"
#endif

#include <stdio.h>

void test1()
{
  Complex a(1.0,2.0), b(3.0,4.0) , c;
  printf("Before calculation\n");
  printf(" a="); a.disp();
  printf(" b="); b.disp();
  printf(" c="); c.disp();
  printf("\n");

  c=a+b*a*Complex(5.0,6.0);
  printf("After c=a+b*a*Complex(5.0,6.0);\n");
  printf(" a="); a.disp();
  printf(" b="); b.disp();
  printf(" c="); c.disp();
  printf("\n");
}

void test2()
{
  Complex a[10];
  int i;
  printf("Array test\n");
  for(i=0;i<10;i++) {
    a[i] = Complex(i,i*2);
    printf(" a[%d]=",i); a[i].disp();
  }
  printf("\n");
}

main()
{
  test1();
  test2();
}
