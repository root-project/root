/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/******************************************************************
* cpp1.cxx
*
* member function call
*
******************************************************************/
#include <stdio.h>
#include <stdlib.h>

int n=0;
int n1=10;

struct b {
  int i;
};

class a {
public:
  int A;
  struct b b;
  void func(void) {printf("a::func() inline A=%d b.i=%d\n",A,b.i);}
  a func1(void);   // optional 
  struct b func2(void);  // optional 
};

a a::func1(void) {
    a result;
    printf("a::func1()\n");
    func() ;
    //result.A ;
    //b;
    printf("b.i=%d\n",b.i);
    //A ;
    printf("A=%d\n",A);
    result.A= ++n;
    result.b.i = -n;
    func();
    printf("func2().i=%d\n",func2().i);
    printf("result.func2().i=%d\n",result.func2().i);
    result.func();
    return(result);
}

struct b a::func2(void)  // member function which returns struct b
{
  printf("a::func2() A=%d b.i=%d\n",A,b.i);
  return(b);
}

class a func(void) {  // in global scope
  class a result;
  printf("func()\n");
  result.A= ++n1;
  result.b.i = -n1;
  return(result);
}

struct a obj;

int main()
{
  class a b;

  printf("a.A=%d\n",func().A);

  obj.A = 1024; 
  obj.b.i = 510;

  b =obj.func1();
  printf ("b.A=%d\n",b.A);

  b=func();
  printf("b.A=%d\n",b.A);

  b.func();

  return 0;
}

