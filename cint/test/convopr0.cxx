/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#ifdef __hpux
#include <iostream.h>
#else
#include <iostream>
using namespace std;
#endif
#include <stdio.h>

class A {
 unsigned int a;
 char *p;
public:
 A(unsigned int ain,char* pin) { a=ain; p=pin; }
 operator unsigned int() { return a;}
 operator char*() { return p;}
};

int main() {
  A x(123,"abcde");
  unsigned int a= (unsigned int)x ;
  char *p = (char*)x;
  for(int i=0;i<5;i++) {
    a = (unsigned int)x;
    p = (char*)x;
    cout <<  p << a << endl;
    printf("<%s>\n",(char*)x);
    cout << "a=" << (unsigned int)x << endl;
    cout << "p=" << (char*)x << endl;
  }
  return 0;
}

