/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <stdio.h>


class A {
 public:
  double d;
  A(double in=0) : d(in) { }
  void disp() const { printf("A::disp() %g\n",d); }
};

namespace FOO{

  class bug{
  private:
    int i;
    
  public:

    bug(int j) : i(j) {}
    int get() { return i; }
    void set(int j) { i=j; }
    friend bug operator+(const bug& rhs, int c);
    void disp() const { printf("FOO::bug::disp() %d\n",i); }
  };
#if 1
  bug operator+(const bug& rhs, int c){
    printf("FOO::operator+(const bug&,int c);\n");
    return bug(rhs.i+c);
  }
#endif

  A operator+(const A& rhs, double c){
    printf("FOO::operator+(const A&,double c);\n");
    return A(rhs.d+c);
  }
  
};

#if 0
FOO::bug operator+(const FOO::bug& rhs, int c){
  printf("::operator+(const FOO::bug&,int c);\n");
  return FOO::bug(rhs.i+c);
}
#endif


//using namespace FOO;
void func() {
  A a;
  //A b = a+3.14;
  
  FOO::bug c(10);
  FOO::bug d = c+10;
  a.disp();
  d.disp();
}

#ifdef __MAKECINT__
#pragma link C++ nestedclasses;
#pragma link C++ nestedtypedef;
#endif

