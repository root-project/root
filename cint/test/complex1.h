/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <stdio.h>

class complex {
 public:
  double re,im;
  complex(double a=0,double b=0) { re=a; im=b; }
  complex& operator =(const complex& a) {
    this->re=a.re;
    this->im=a.im;
    return(*this);
  }
#ifdef NEVER
  compelx(const complex& a) { re=a.re; im=a.im; }
  ~complex() { }
#endif
  complex& operator()(int i,int j);
  double& operator[](int i);
};

int operator<(const complex& a,const complex& b);
int operator<=(const complex& a,const complex& b);
int operator>(const complex& a,const complex& b);
int operator>=(const complex& a,const complex& b);
int operator==(const complex& a,const complex& b);
int operator!=(const complex& a,const complex& b);

complex operator+(const complex& a,const complex& b);
complex operator-(const complex& a,const complex& b);
complex operator*(const complex& a,const complex& b);
complex operator/(const complex& a,const complex& b);

complex exp(const complex& a);
double abs(const complex& a);
double re(const complex& a);
double im(const complex& a);

typedef unsigned int uint32;
extern uint32 uint32obj;

enum fruits { orange, apple, others };

class A {
  int i;
  double d;
 public:
  int I() { return i; }
  int I(int in) { i=in; return(i); }
  double D(void) { return(d); }
  double D(double in) { d=in; return(d); }

  unsigned int b0 : 1;
  unsigned int b1 : 1;
  unsigned int b2 : 1;
};

extern A AOBJ;

const int cf1(const int a);
const int& cf2(const int& a);
int const& cf3(int const& a);
int const & cf4(int const & a);
int const &cf5(int const &a);
const int & cf6(const int & a);
const int* cf7(const int* a);
const int *const cf8(const int *const a);
const int * const cf9(const int * const a);

void dop2f(int (*f)(int),int& a);

#ifdef __MAKECINT__
double macrofunc(float a,short b);
#else
#define macrofunc(a,b) (a*b)
#endif


