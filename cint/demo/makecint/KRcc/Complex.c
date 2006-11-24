/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/**************************************************************************
* Complex.c
*
*
**************************************************************************/

#include "Complex.h"

/* Source file can have K&R style function header only. Makecint or cint -c-2
* does not look int source files for making interface method.
*/

struct Complex j ;

void ComplexInit()
{
  j.re=0;
  j.im=1;
}

void ComplexSet(pa,rein,imin)
struct Complex *pa;
double rein;
double imin;
{
	pa->re=rein;
	pa->im=imin;
}

struct Complex ComplexAdd(a,b)
struct Complex a;
struct Complex b;
{
	struct Complex c;
	c.re = add(a.re,b.re);
	c.im = add(a.im,b.im);
	/* c.re = a.re+b.re; */
	/* c.im = a.im+b.im; */
	return(c);
}

struct Complex ComplexMultiply(a,b)
struct Complex a;
struct Complex b;
{
	struct Complex c;
	c.re = a.re*b.re-a.im*b.im;
	c.im = a.re*b.im+a.im*b.re;
	return(c);
}


/***********************************************************************/

struct Complex ComplexExp(a)
struct Complex a;
{
	struct Complex c;
	double mag;
	mag = exp(a.re);
	c.re=mag*cos(a.im);
	c.im=mag*sin(a.im);
	return(c);
}

double ComplexAbs(a)
struct Complex a;
{
	double result;
	result = sqrt(a.re*a.re+a.im*a.im);
	return(result);
}

double ComplexReal(a)
struct Complex a;
{
	return(a.re);
}

double ComplexImag(a)
struct Complex a;
{
	return(a.im);
}

void ComplexDisplay(a)
struct Complex a;
{
	Display(a);
}
