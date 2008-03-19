/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/**************************************************************************
* Complex.C
*
*
**************************************************************************/

#include "Complex.h"

void Complex::disp(void)
{
  printf("(%g,%g)",re,im);
}


Complex operator +(Complex const & a,Complex const & b)
{
	Complex c;
	c.re = a.re+b.re;
	c.im = a.im+b.im;
	return(c);
}

Complex operator -(Complex const & a,Complex const & b)
{
	Complex c;
	c.re = a.re-b.re;
	c.im = a.im-b.im;
	return(c);
}

Complex operator *(Complex const & a,Complex const & b)
{
	Complex c;
	c.re = a.re*b.re-a.im*b.im;
	c.im = a.re*b.im+a.im*b.re;
	return(c);
}

Complex operator /(Complex const & a,Complex const & b)
{
	Complex c;
	double x;
	x = b.re*b.re+b.im*b.im;
	c.re = (a.re*b.re+a.im*b.im)/x;
	c.im = (a.im*b.re-a.re*b.im)/x;
	return(c);
}

//**********************************************************************

Complex exp(Complex& a)
{
	Complex c;
	double mag;
	mag = exp(a.re);
	c.re=mag*cos(a.im);
	c.im=mag*sin(a.im);
	return(c);
}

double fabs(Complex& a)
{
	double result;
	result = sqrt(a.re*a.re+a.im*a.im);
	return(result);
}

double real(Complex& a)
{
	return(a.re);
}

double imag(Complex& a)
{
	return(a.im);
}



//**********************************************************************
ostream& operator<<(ostream& ios,Complex& a)
{
  ios << '(' << a.real() << ',' << a.imag() << ')' ;
  ios.flush();
  return(ios);
}



// Added for test
const int cf1(const int a){return(a);}
const int& cf2(const int& a){return(a);}
int const& cf3(int const& a){return(a);}
int const & cf4(int const & a){return(a);}
int const & cf5(int const & a){return(a);}
const int & cf6(const int & a){return(a);}
const int* cf7(const int* a){return(a);}
const int *const cf8(const int *const a){return(a);}
const int * const cf9(const int * const a){return(a);}
