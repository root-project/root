/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/**************************************************************************
* array.c
*
* Simple Array class
*
*  Constructor, copy constructor, destructor, operator overloading
* function overloading, reference type
*
**************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int nstore;

/**********************************************************
* class array
**********************************************************/

struct array  {
  double *dat;
  int n;

  array(array& X);
  array(double x,int ndef);
  array(double x);
  ~array(); 
  void print(void);
  void print(int n);
  array operator =(array& a);
} ;

array::~array()
{
  // abc
  free(dat);
  // printf("destroyed\n");
}

// Copy constructor
array::array(array& X)
{
  int i;
  // free(dat);
  dat = (double *)malloc(X.n*sizeof(double));
  for(i=0;i<X.n;i++) dat[i] = X.dat[i];
  n = X.n;
}

// constructor
array::array(double x,int ndef)
{
  dat = (double *)malloc(ndef*sizeof(double));
  for(n=0;n<ndef;n++) dat[n] = x ;
  n = ndef;
  nstore = n;
}

// constructor for implicit conversion
array::array(double x)
{
  if(nstore==0) {
    fprintf(stderr,"Error: Size of array 0\n");
    return;
  }
  dat = (double *)malloc(nstore*sizeof(double));
  for(n=0;n<nstore;n++) dat[n] = x ;
  n=nstore;
}

void array::print(void)
{
  int i;
  for(i=0;i<n;i++) {
    printf("%g ",dat[i]);
  }
  printf("\n");
}

void array::print(int n)
{
  int i;
  for(i=0;i<n;i++) {
    printf("%g ",dat[i]);
  }
  printf("\n");
}

/**********************************************************
* class array operator overloading
**********************************************************/

array array::operator =(array& a)
{
  int i;
  free(dat);
  dat = (double *)malloc(a.n*sizeof(double));
  for(i=0;i<a.n;i++) dat[i]=a.dat[i];
  n=a.n;
  return(*this);
}

array operator +(array& a,array& b)
{
  array c(0);
  int i;
  for(i=0;i<nstore;i++) {c.dat[i] = a.dat[i]+b.dat[i] ;}
  c.n=a.n;
  return(c);
}

array operator -(array& a,array& b)
{
  array c(0);
  int i;
  for(i=0;i<nstore;i++) {c.dat[i] = a.dat[i]-b.dat[i] ;}
  c.n=a.n;
  return(c);
}

array operator *(array& a,array& b)
{
  array c(0);
  int i;
  for(i=0;i<nstore;i++) {c.dat[i] = a.dat[i]*b.dat[i] ;}
  c.n=a.n;
  return(c);
}

array operator <<(array& a,int shift)
{
  array c(0);
  int i;
  for(i=0;i<a.n-shift;i++) {c.dat[i] = a.dat[i+shift] ;}
  c.n=a.n;
  return(c);
}

/**********************************************************
* class array function overloading
**********************************************************/

array sin(array& a)
{
  array c(0);
  int i;
  for(i=0;i<a.n;i++) c.dat[i] = sin(a.dat[i]);
  c.n=a.n;
  return(c);
}

array cos(array& a)
{
  array c(0);
  int i;
  for(i=0;i<a.n;i++) c.dat[i] = cos(a.dat[i]);
  c.n=a.n;
  return(c);
}



/**********************************************************
* Incredible main routine!!!  This simple program does all
* the array calculation. But if you trace it, it's a mess.
* 'cint -t cpp6.c' will make you sick.
**********************************************************/

const int size=10;
const double PI=3.141592;
array a=array(0,size),b=array(0);

main()
{
  int i;
  array c=array(0);

  for(i=0;i<size;i++) {
    a.dat[i] = i;
    b.dat[i] = i*2;
  }

  a.print();
  b.print();

  c=a+b;
  c.print();

  c=a*b+c;
  c.print();

  c=a+b*5-10;
  c.print();

  c=(c<<2)-12;
  c.print();

  c=(sin(a*PI)*cos(b*PI))*1e7*sin(PI*0.25)+a;
  c.print();
}
