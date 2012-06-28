/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <stdio.h>
class POINT{
public:
  float operator[](int);
  float value(int);
  float value1(int);
};

float POINT::operator[](int i) { return i*1.2; }
float POINT::value(int i) { return (*this)[i]; }

void test1() {
  POINT a;
  printf("%g\n",a.value(1));
}


void test2() {
  POINT a;
  for(int i=0;i<5;i++) {
    printf("%g\n",a.value(i));
  }
}


int main() {
  test1();
  test2();
  return 0;
}

