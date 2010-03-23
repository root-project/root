/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <stdio.h>

typedef long Long_t;
typedef unsigned long ULong_t;
typedef long long Long64_t;
typedef unsigned long long ULong64_t;
typedef long double Double92_t;
typedef double Double_t;

void test0() {
#if 0
  ULong_t a = dynamic_cast<ULong_t>(123);
  ULong64_t b = dynamic_cast<ULong64_t>(456);
  Double92_t c = dynamic_cast<Double92_t>(1.234);
#else
  ULong_t a = (ULong_t)(123);
  ULong64_t b = (ULong64_t)(456);
  Double92_t c = (Double92_t)(1.234);
  Double_t d = (Double_t)(5.678);
#endif
  printf("%lu %llu %LG %g\n",a,b,c,d);
  printf("%lu  %g\n",a,d);
  printf("%LG\n",c);
}

void test1() {
  printf("test1\n");
  ULong64_t aa;
  aa = (ULong64_t)(-(1<<30 - 1));
  printf("1: %llu ",aa);
  aa = aa<<1;
  printf("2: %llu ",aa);
  aa = aa*2;
  printf("3: %llu ",aa);
  ULong64_t bb = aa;
  // printf("4-0: %x %x ",1u<<30 - 1,-(1u<<30 - 1));
  // Long_t ii = -(1u<<28-1); // -(536870913u-1); // 1u<<30 - 1);
  // printf("4-1: %lld %llu %llu ",ii,(ULong64_t)ii, aa);
  // aa = bb / ii;
  aa = bb/((Long64_t)(-(1u<<30 - 1)));
  printf("4: %llu\n",aa);
}

void test2() {
  printf("test2\n");
  ULong64_t aa;
  aa = 1ULL<<31;
  printf("1: %llu ",aa);
  aa = aa<<1;
  printf("2: %llu ",aa);
  aa = aa*2;
  printf("3: %llu ",aa);
  aa = aa/((Long64_t)(1u<<31));
  printf("4: %llu\n",aa);
}

void test3() { 
  printf("test3\n");
 Long64_t aa;
  aa = (int)(1u<<31);
  printf("1: %lld ",aa);
  aa = aa<<1;
  printf("2: %lld ",aa);
  aa = aa*2;
  printf("3: %lld ",aa);
  aa = aa/(1LL<<31);
  printf("4: %lld\n",aa);
}

void test4() {
  printf("test4\n");
  Long64_t aa;
  aa = 1LL<<31;
  printf("1: %lld ",aa);
  aa = aa<<1;
  printf("2: %lld ",aa);
  aa = aa*2;
  printf("3: %lld ",aa);
  aa = aa/(1LL<<31);
  printf("4: %lld\n",aa);
}

int main() {
  test0();
  test1();
  test2();
  test3();
  test4();
  return 0;
}
