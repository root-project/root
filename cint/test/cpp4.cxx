/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/**************************************************************************
* cpp4.c
*
* function, operator overloading 
* and implicit type conversion
*
**************************************************************************/
#include <stdlib.h>
#include <stdio.h>


/**********************************************************
* class complex
**********************************************************/

class complex  {
 public:
  double re,im;

  //complex(double re=2,double im=4); // TODO, problem if argument name
  // differ between header and function definition.
  complex(double real=2,double imag=4);
  void print(void) { printf("re=%g , im=%g\n",re,im); }

} ;

complex::complex(double real,double imag)
{
  re=real;
  im=imag;
}

complex operator + (complex a, complex b)
{
  complex c;
  c.re = a.re + b.re ;
  c.im = a.im + b.im ;
  return(c);
}
/* Done by implicit conversion
complex operator + (double a, complex b)
{
  complex c;
  c.re = a + b.re ;
  c.im = b.im ;
  return(c);
}
complex operator + (complex a, double b)
{
  complex c;
  c.re = a.re + b ;
  c.im = a.im ;
  return(c);
}
*/

complex operator - (complex a, complex b)
{
  complex c;
  c.re = a.re - b.re ;
  c.im = a.im - b.im ;
  return(c);
}
complex operator - (double a, complex b)
{
  complex c;
  c.re = a - b.re ;
  c.im = - b.im ;
  return(c);
}
complex operator - (complex a, double b)
{
  complex c;
  c.re = a.re - b ;
  c.im = a.im ;
  return(c);
}

complex operator * (complex a, complex b)
{
  complex c;
  c.re = a.re * b.re - a.im * b.im ;
  c.im = a.re * b.im + a.im * b.re ;
  return(c);
}
complex operator * (double a, complex b)
{
  complex c;
  c.re = a * b.re;
  c.im = a * b.im;
  return(c);
}
complex operator * (complex a, double b)
{
  complex c;
  c.re = a.re * b ;
  c.im = a.im * b;
  return(c);
}

complex plus(complex a,complex b)
{
  complex c;
  c.re = a.re + b.re ;
  c.im = a.im + b.im ;
  return(c);
}



/**********************************************************
* main program
**********************************************************/
const complex j(0,1);

int main () 
{
  complex a,b(3,4),c;

  a.re = 1 ;
  a.im = 2 ;
  //b.re = 3;
  //b.im = 4;

  c = a + b;
  c.print();

  c = a*5;
  c.print();

  c = 30*a;
  c.print();

  c = a * b;
  c.print();

  c = a + b - j*3;
  c.print();

  c = a + 3;  // problem
  c.print();

  c = plus(a,b);
  c.print();

  c = plus(a,7);
  c.print();

  c = plus(15,b);
  c.print();

  // c = 51;
  // c.print();

  return 0;
}

