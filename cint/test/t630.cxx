/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>
//#include <iostream>

//#define TEST

class Foo {
public:
  Foo(int i) { 
#ifdef TEST
    printf("foo-ctor: %d\n",i);
#endif
    _i = i; 
  }
  ~Foo()     {
#ifdef TEST
    printf("foo-dtor: %d\n",_i); 
#endif
  }
  //int operator==(Foo& other){ return(_i==other._i); }
  int operator==(Foo other){ return(_i==other._i); }
private:
  int _i;
};

void f()
{
  Foo f1=7;
  // '3' will or will not be turned into a temporary Foo object
  bool b1 = ( 1 && f1==3 ); // temp object c'tor + d'tor
  bool b2 = ( 0 && f1==3 );   // temp object d'tor only!
  if(b1==b2) printf("match\n");
  else       printf("unmatch\n");
}

void g()
{
  Foo f1=2;
  bool b1,b2;
  int i;
  // '3' will or will not be turned into a temporary Foo object
  for(i =0;i<5;i++) {
    b1 = ( 1 && f1==i ); // temp object c'tor + d'tor
    b2 = ( 0 && f1==i );   // temp object d'tor only!
    if(b1==b2) printf("match\n");
    else       printf("unmatch\n");
  }
}

int main() {
  f();
  g();
  return 0;
}
