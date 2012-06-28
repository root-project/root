/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/**************************************************************************
* cpp0.cxx
*
* operator overloading, default parameter, constructor
*
* This probram doesn't run correctly on cint.
* Complex object is included as member in carray. Default constructor of
* complex must be called when carray is created.
*
**************************************************************************/
#include <stdlib.h>
#include <stdio.h>

struct a {
  int i;
  void inc() { i++; }
};


struct a b;

/*
a::inc()
{
  int i;
  i++;
}
*/

/**********************************************************
* class complex
**********************************************************/

struct  complex{
 public:
  double re,im;
  complex(double real=2,double imag=4);
};

complex::complex(double real,double imag)
{
  re=real;
  im=imag;
  // printf("%g %g\n",2,4);
}

complex operator + (complex a, complex b)
{
  complex c;
  c.re = a.re + b.re ;
  c.im = a.im + b.im ;
  return(c);
}

complex operator - (complex a, complex b)
{
  complex c;
  c.re = a.re - b.re ;
  c.im = a.im - b.im ;
  return(c);
}

complex operator * (complex a, complex b)
{
  complex c;
  c.re = a.re * b.re - a.im * b.im ;
  c.im = a.re * b.im + a.im * b.re ;
  return(c);
}

complex operator & (complex a, complex b)
{
  complex c;
  c.re = a.re + b.re;
  c.im = a.im + b.im;
  return(c);
}


/**********************************************************
* class carray
**********************************************************/
class carray {
 public:
  int n;
  complex dat[100];
  carray(double re=0,double im=0);
} ;

carray::carray(double re,double im)
{
  int i;
  for(i=0;i<100;i++) {
    dat[i].re = re;
    dat[i].im = im;
  }
}

carray operator + (carray a, carray b)
{
  int i;
  carray c;
  for(i=0;i<100;i++) {
    c.dat[i].re = a.dat[i].re + b.dat[i].re ;
    c.dat[i].im = a.dat[i].im + b.dat[i].im ;
  }
  return(c);
}

carray operator - (carray a, carray b)
{
  int i;
  carray c;
  for(i=0;i<100;i++) {
    c.dat[i].re = a.dat[i].re - b.dat[i].re ;
    c.dat[i].im = a.dat[i].im - b.dat[i].im ;
  }
  return(c);
}

carray operator * (carray a, carray b)
{
  int i;
  carray c;
  for(i=0;i<100;i++) {
    c.dat[i].re = a.dat[i].re*b.dat[i].re - a.dat[i].im*b.dat[i].im;
    c.dat[i].im = a.dat[i].re*b.dat[i].im + a.dat[i].im*b.dat[i].re;
  }
  return(c);
}



/**********************************************************
* main program
**********************************************************/
carray d,e(1),f(2,3);

int main () 
{
  complex a,b(3,4),c;
  int i;

  a.re = 1 ;
  a.im = 2 ;
  //b.re = 3;
  //b.im = 4;
  c = a + b;
  printf("c.re = %g  c.im = %g \n",c.re,c.im);

  c = a * b;
  printf("c.re = %g  c.im = %g \n",c.re,c.im);

  for(i=0;i<100;i++) {
    d.dat[i].re = 1*i;
    d.dat[i].im = 2*i;
    e.dat[i].re = 3*i;
    e.dat[i].im = 4*i;
  }

  f = d + e;
  f = (d + e) + (d-e) ;
  
  for(i=0;i<10;i++) {
    printf("%d %5g %5g %5g %5g %5g %5g\n",i
	   ,d.dat[i].re,d.dat[i].im
	   ,e.dat[i].re,e.dat[i].im
	   ,f.dat[i].re,f.dat[i].im);
  }

  return 0;
}


int ansi(int a,double b)
{
  return (a*(int)b);
}

/*
standard(a,b)
int a;
double b;
{
}
*/
