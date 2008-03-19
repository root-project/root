/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/**************************************************************************
* cpp5.cxx
*
*  operator overloading for array type
*  operator=() overloading
*
**************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


/**********************************************************
* class complex
**********************************************************/
int nstore;

class array  {
 public:
  double dat[100];
  int n;

 public:
  array(double x);
  array(double x,int ndef);
  ~array() ;

  void print(void);

  array operator=(array a);
  array operator<<(int shift);
  array operator>>(int a);
  array operator()(int a, int b);
} ;

array::~array()
{
  // free(dat);
}

array::array(double x,int ndef)
{
  // dat = malloc(ndef*sizeof(double));
  for(n=0;n<ndef;n++) dat[n] = x ;
  n = ndef;
  nstore = n;
}

array::array(double x)
{
  if(nstore==0) {
    fprintf(stderr,"Error: Size of array 0\n");
    return;
  }
  // dat = malloc(nstore*sizeof(double));
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


array operator+(array a,array b)
{
  array c(0);
  int i;
  for(i=0;i<nstore;i++) {c.dat[i] = a.dat[i]+b.dat[i] ;}
  return(c);
}

array operator-(array a,array b)
{
  array c(0);
  int i;
  for(i=0;i<nstore;i++) {c.dat[i] = a.dat[i]-b.dat[i] ;}
  return(c);
}

array operator*(array a,array b)
{
  array c(0);
  int i;
  for(i=0;i<nstore;i++) {c.dat[i] = a.dat[i]*b.dat[i] ;}
  return(c);
}

/* because of no reference type, this is the only way now */
array array::operator=(array a)
{
  int i;
  for(i=0;i<a.n;i++) {dat[i] = a.dat[i] ;}
  n=a.n;
  return(*this);
}


/*
array operator<<(array a,int shift)
{
  array c(0,a.n);
  int i;
  for(i=0;i<a.n-shift;i++) {c.dat[i] = a.dat[i+shift] ;}
  return(c);
}
*/
array array::operator<<(int shift)
{
  array c(0,nstore);
  int i;
  for(i=0;i<n-shift;i++) {c.dat[i] = dat[i+shift] ;}
  return(c);
}

array array::operator()(int a,int b)
{
  array c(0,b-a+1);
  int i;
  for(i=0;i<nstore;i++) {c.dat[i] = dat[a+i];}
  return(c);
}


int main()
{
  array a(1,10),b(2),c(0);

  c=a+b;
  c.print();

  c=a*b+c;
  c.print();

  c=a-b*5+10;
  c.print();

  c=c<<2;
  c.print();

  return 0;
}

