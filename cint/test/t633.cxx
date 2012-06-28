/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#if defined(interp) && defined(makecint)
#pragma include "test.dll"
#else
#include "t633.h"
#endif


void test1() {
  printf("test1\n");
  B b;
  A &chip = b.Get(1);
  printf("x chip.a=%d  &chip=%d\n",chip.a,chip.id);
  int i;
  for(i=0;i<3;i++) {
    A &chip = b.Get(i);
    printf("4 chip.a=%d  &chip=%d\n",chip.a,chip.id);
  }
}

void test2() {
  printf("test2\n");
  B b;
  int i;
  for(i=0;i<3;i++) {
    A &chip = b.Get(i);
    printf("4 chip.a=%d  &chip=%d\n",chip.a,chip.id);
  }
}

void test3() {
  printf("test3\n");
  B b;
  int i;
  for(i=0;i<3;i++) {
    A &chip = b.Get(i);
    printf("1 chip.a=%d  &chip=%d\n",chip.a,chip.id);
    chip.a = i+100;
    printf("2 chip.a=%d  &chip=%d\n",chip.a,chip.id);
  }
  for(i=0;i<3;i++) {
    A &chip = b.Get(i);
    printf("4 chip.a=%d  &chip=%d\n",chip.a,chip.id);
  }
}


void test4() {
  printf("test4\n");
  B b;
  A chip = b.Get(1);
  printf("x chip.a=%d  &chip=%d\n",chip.a,chip.id);
  int i;
  for(i=0;i<3;i++) {
    A chip = b.Get(i);
    printf("4 chip.a=%d  &chip=%d\n",chip.a,chip.id);
  }
}

void test5() {
  printf("test5\n");
  B b;
  int i;
  for(i=0;i<3;i++) {
    A chip = b.Get(i);
    printf("4 chip.a=%d  &chip=%d\n",chip.a,chip.id);
  }
}

void test6() {
  printf("test6\n");
  B b;
  int i;
  for(i=0;i<3;i++) {
    A chip = b.Get(i);
    printf("1 chip.a=%d  &chip=%d\n",chip.a,chip.id);
    chip.a = i+100;
    printf("2 chip.a=%d  &chip=%d\n",chip.a,chip.id);
  }
  for(i=0;i<3;i++) {
    A chip = b.Get(i);
    printf("4 chip.a=%d  &chip=%d\n",chip.a,chip.id);
  }
}

int main() {
#ifdef DEBUG
  test4();
#else
  test1();
  test2();
  test3();
#ifdef TEST
  test4();
  test5();
  test6();
#endif
#endif
  return 0;
}
