/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/**************************************************************************
* complex.C
*
*
**************************************************************************/
#include "complex1.h"
#include <math.h>



#ifdef NEVER
void *complex::operator new(int i)
{
}

void complex::operator delete(void *p)
{
}
#endif

complex& complex::operator()(int i,int j) 
{ 
	// printf("operator()(%d,%d)\n",i,j); 
	re+=i;
	im+=j;
	return(*this); 
} 

double& complex::operator[](int i) 
{ 
	// printf("operator[](%d)\n",i); 
	if(i==0) return(re); 
	else     return(im);
}


//

int operator<(const complex& a,const complex& b)
{
	return(abs(a)<abs(b));
}

int operator<=(const complex& a,const complex& b)
{
	return(abs(a)<=abs(b));
}

int operator>(const complex& a,const complex& b)
{
	return(abs(a)>abs(b));
}

int operator>=(const complex& a,const complex& b)
{
	return(abs(a)>=abs(b));
}

int operator==(const complex& a,const complex& b)
{
	return(abs(a)==abs(b));
}

int operator!=(const complex& a,const complex& b)
{
	return(abs(a)!=abs(b));
}


//

complex operator +(const complex& a,const complex& b)
{
	complex c(0);
	c.re = a.re+b.re;
	c.im = a.im+b.im;
	return(c);
}

complex operator -(const complex& a,const complex& b)
{
	complex c(0);
	c.re = a.re-b.re;
	c.im = a.im-b.im;
	return(c);
}

complex operator *(const complex& a,const complex& b)
{
	complex c(0);
	c.re = a.re*b.re-a.im*b.im;
	c.im = a.re*b.im+a.im*b.re;
	return(c);
}

complex operator /(const complex& a,const complex& b)
{
	complex c(0);
	double x;
	x = b.re*b.re+b.im*b.im;
	c.re = (a.re*b.re+a.im*b.im)/x;
	c.im = (a.im*b.re-a.re*b.im)/x;
	return(c);
}


//

complex exp(const complex& a)
{
	complex c(0);
	double mag;
	mag = exp(a.re);
	c.re=mag*cos(a.im);
	c.im=mag*sin(a.im);
	return(c);
}

double abs(const complex& a)
{
	double result;
	result = sqrt(a.re*a.re+a.im*a.im);
	return(result);
}

double re(const complex& a)
{
	return(a.re);
}

double im(const complex& a)
{
	return(a.im);
}


//

uint32 uint32obj=15;
A AOBJ;


const int cf1(const int a){ printf("cf1 %d\n",a); return(a);}
const int& cf2(const int& a){ printf("cf2 %d\n",a); return(a);}
int const& cf3(int const& a){ printf("cf3 %d\n",a); return(a);}
int const & cf4(int const & a){ printf("cf4 %d\n",a); return(a);}
int const & cf5(int const & a){ printf("cf5 %d\n",a); return(a);}
const int & cf6(const int & a){ printf("cf6 %d\n",a); return(a);}
const int* cf7(const int* a){ printf("cf7 %ld\n",(long)a); return(a);}
const int *const cf8(const int *const a){ printf("cf8 %ld\n",(long)a);return(a); }
const int * const cf9(const int * const a){ printf("cf9 %ld\n",(long)a); return(a);}

void dop2f(int (*f)(int),int& a) {
  f(a);
}

