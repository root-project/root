/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/**************************************************************************
* darray.C
*
* Array class
*
*  Constructor, copy constructor, destructor, operator overloading
* function overloading, reference type
*
**************************************************************************/

/**********************************************************
* definition of array class
**********************************************************/
#include "darray.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


void G__ary_assign(double *c,double start,double stop,int n)
{
  int i;
  double res;
  res = (stop-start)/(n-1);
  for(i=0;i<n;i++) c[i] = i*res + start ;
}

void G__ary_plus(double *c,double *a,double *b,n)
  double *c,*a,*b;
int n;
{
  int i;
  for(i=0;i<n;i++) c[i] = a[i] + b[i];
}

void G__ary_minus(c,a,b,n)
double *c,*a,*b;
int n;
{
  int i;
  for(i=0;i<n;i++) c[i] = a[i] - b[i];
}

void G__ary_multiply(c,a,b,n)
double *c,*a,*b;
int n;
{
  int i;
  for(i=0;i<n;i++) c[i] = a[i] * b[i];
}

void G__ary_divide(c,a,b,n)
double *c,*a,*b;
int n;
{
  int i;
  for(i=0;i<n;i++) {
    if(b[i]!=0) c[i] = a[i] / b[i];
    else {
      if(a[i]==0) c[i]=1;
      else if(a[i]>0) c[i]=1e101;
      else c[i]=-1e101;
    }
  }
}

void G__ary_power(c,a,b,n)
double *c,*a,*b;
int n;
{
  int i,j;
  int flag=0;
  double r;
  for(i=0;i<n;i++) {
    if(a[i]>0) c[i] = exp(b[i]*log(a[i]));
    else if(a[i]==0) c[i]=0;
    else { 
      if(fmod(b[i],1.0)==0.0) {
	r=1;
	for(j=0;j<(int)a[i];j++) 
	  r *= b[i];
      }
      else if(flag==0) {
	fprintf(stderr,"Error: Power operator oprand<0\n");
	flag++;
      }
    }
  }
}

void G__ary_exp(c,a,n)
  double *c,*a;
int n;
{
  int i;
  for(i=0;i<n;i++) c[i] = exp(a[i]);
}

void G__ary_log(c,a,n)
  double *c,*a;
int n;
{
  int i;
  for(i=0;i<n;i++) {
    if(a[i]>0) c[i] = log(a[i]);
    else       c[i] = -1e101;
  }
}

void G__ary_log10(c,a,n)
  double *c,*a;
int n;
{
  int i;
  for(i=0;i<n;i++) {
    if(a[i]>0) c[i] = log10(a[i]);
    else       c[i] = -1e101;
  }
}

void G__ary_sinc(c,a,n)
  double *c,*a;
int n;
{
  int i;
  for(i=0;i<n;i++) {
    if(a[i]!=0) c[i] = sin(a[i])/a[i];
    else        c[i] = 1;
  }
}

void G__ary_sin(c,a,n)
  double *c,*a;
int n;
{
  int i;
  for(i=0;i<n;i++) c[i] = sin(a[i]);
}

void G__ary_cos(c,a,n)
  double *c,*a;
int n;
{
  int i;
  for(i=0;i<n;i++) c[i] = cos(a[i]);
}

void G__ary_tan(c,a,n)
  double *c,*a;
int n;
{
  int i;
  for(i=0;i<n;i++) c[i] = tan(a[i]);
}

void G__ary_sinh(c,a,n)
  double *c,*a;
int n;
{
  int i;
  for(i=0;i<n;i++) c[i] = sinh(a[i]);
}

void G__ary_cosh(c,a,n)
  double *c,*a;
int n;
{
  int i;
  for(i=0;i<n;i++) c[i] = cosh(a[i]);
}

void G__ary_tanh(c,a,n)
  double *c,*a;
int n;
{
  int i;
  for(i=0;i<n;i++) c[i] = tanh(a[i]);
}


void G__ary_asin(c,a,n)
  double *c,*a;
int n;
{
  int i;
  for(i=0;i<n;i++) c[i] = asin(a[i]);
}

void G__ary_acos(c,a,n)
  double *c,*a;
int n;
{
  int i;
  for(i=0;i<n;i++) c[i] = acos(a[i]);
}

void G__ary_atan(c,a,n)
  double *c,*a;
int n;
{
  int i;
  for(i=0;i<n;i++) c[i] = atan(a[i]);
}

void G__ary_fabs(c,a,n)
  double *c,*a;
int n;
{
  int i;
  for(i=0;i<n;i++) c[i] = fabs(a[i]);
}

void G__ary_sqrt(c,a,n)
  double *c,*a;
int n;
{
  int i;
  for(i=0;i<n;i++) c[i] = sqrt(a[i]);
}


void G__ary_rect(c,a,n)
  double *c,*a;
int n;
{
  int i;
  for(i=0;i<n;i++) {
    if(-0.5<a[i] && a[i]<0.5) 
      c[i]=1.0;
    else if(a[i]==0.5 || a[i]==0.5)
      c[i]=0.5;
    else
      c[i]=0.0;
  }
}

void G__ary_square(c,a,n)
  double *c,*a;
int n;
{
  int i;
  double tmp;
  for(i=0;i<n;i++) {
    tmp = sin(a[i]);
    if(tmp<0)       c[i] = -1.0;
    else if(tmp==0) c[i]=   0.0;
    else            c[i] =  1.0;
  }
}

void G__ary_rand(c,a,n)
  double *c,*a;
int n;
{
  int i;
  for(i=0;i<n;i++) {
    c[i] = drand48();
  }
}

void G__ary_conv(c,a,n,b,m)
  double *c,*a,*b;
int n,m;
{
  int i,j,k;
  int f,t;
  f = m/2;
  t = m-f;
  for(i=0;i<n;i++) {
    c[i]=0.0;
    for(j=0;j<m;j++) {
      k=i-f+j;
      if(k<0)       c[i] += a[0]*b[j];
      else if(k>=n) c[i] += a[n-1]*b[j];
      else          c[i] += a[k]*b[j];
    }
  }
}

void G__ary_integ(c,a,b,n)
  double *c,*a,*b;
int n;
{
  int i;
  double integ=0.0;
  for(i=0;i<n-1;i++) {
    integ += b[i]*(a[i+1]-a[i]);
    c[i] = integ;
  }
  integ += b[i]*(a[i]-a[i-1]);
  c[i] = integ;
}

void G__ary_diff(c,a,b,n)
  double *c,*a,*b;
int n;
{
  int i;
  double integ=0.0;
  for(i=0;i<n;i++) {
    c[i] = (b[i+1]-b[i])/(a[i+1]-a[i]);
  }
  c[i] = c[i-1];
}




/***********************************************
* Destructor
***********************************************/
array::~array()
{
  delete[] dat;
}

/***********************************************
* Copy constructor
***********************************************/
array::array(array& X)
{
  int i;
  dat = new double[X.n];
  memcpy(dat,X.dat,X.n*sizeof(double));
  n = X.n;
}

/***********************************************
* Implicit conversion constructor 
***********************************************/
array::array(double x)
{
  if(G__arraysize==0) {
    cerr << "Error: Size of array 0\n";
    return;
  }
  dat = new double[G__arraysize];
  G__ary_assign(dat,x,x,G__arraysize);
  n=G__arraysize;
}


/***********************************************
* Constructor
***********************************************/
array::array(double start,double stop,int ndat)
{
  double res;
  G__arraysize=ndat;
  dat = new double[G__arraysize];
  G__ary_assign(dat,start,stop,G__arraysize);
  n = G__arraysize;
}

/***********************************************
* constructor 
***********************************************/
array::array(void)
{
  if(G__arraysize==0) {
    cerr << "Error: Size of array 0\n";
    return;
  }
  dat = new double[G__arraysize];
  n=G__arraysize;
}

/***********************************************
* constructor 
***********************************************/
array::array(double *p,int ndat)
{
  G__arraysize=ndat;
  dat = new double[G__arraysize];
  memcpy(dat,p,ndat*sizeof(double));
  n = G__arraysize;
}


/***********************************************
* constructor for rvalue subarray
***********************************************/
array::array(array& X,int offset,int ndat)
{
  int i;
  dat = new double[ndat];
  if(offset+ndat>X.n) {
    memcpy(dat,X.dat+offset,(X.n-offset)*sizeof(double));
    for(i=X.n-offset;i<ndat;i++) dat[i] = 0.0;
  }
  else {
    memcpy(dat,X.dat+offset,ndat*sizeof(double));
  }
  n = ndat;
}

/***********************************************
* resize
***********************************************/
int array::resize(int size)
{
  double *temp;

  if(size!=n) {
    delete[] dat;
    dat=new double[size];
    n=size;
  }
  return(n);
}

/**********************************************************
* operator = as member function
**********************************************************/
array& array::operator =(array& a)
{
  int i;
  if(a.n<n) memcpy(dat,a.dat,a.n*sizeof(double));
  else      memcpy(dat,a.dat,n*sizeof(double));
  return(*this);
}


/***********************************************
* operator +
***********************************************/
array operator +(array& a,array& b)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_plus(c.dat,a.dat,b.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* operator -
***********************************************/
array operator -(array& a,array& b)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_minus(c.dat,a.dat,b.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* operator *
***********************************************/
array operator *(array& a,array& b)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_multiply(c.dat,a.dat,b.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* operator /
***********************************************/
array operator /(array& a,array& b)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_divide(c.dat,a.dat,b.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* operator @ (power)
***********************************************/
array operator @(array& a,array& b)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_power(c.dat,a.dat,b.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* operator << (shift)
***********************************************/
array operator <<(array& a,int shift)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  for(i=0;i<a.n-shift;i++) {c.dat[i] = a.dat[i+shift] ;}
  c.n=a.n;
  return(c);
}

/***********************************************
* operator >> (shift)
***********************************************/
array operator >>(array& a,int shift)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  for(i=0;i<a.n-shift;i++) {c.dat[i+shift] = a.dat[i] ;}
  c.n=a.n;
  return(c);
}


/**********************************************************
* class array function overloading
**********************************************************/

/***********************************************
* exp
***********************************************/
array exp(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_exp(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* log
***********************************************/
array log(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_log(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* log10
***********************************************/
array log10(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_log10(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* sinc
***********************************************/
array sinc(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_sinc(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* sin
***********************************************/
array sin(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_sin(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* cos
***********************************************/
array cos(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_cos(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* tan
***********************************************/
array tan(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_tan(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* sinh
***********************************************/
array sinh(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_sinh(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* cosh
***********************************************/
array cosh(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_cosh(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* tanh
***********************************************/
array tanh(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_tanh(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* asin
***********************************************/
array asin(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_asin(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* acos
***********************************************/
array acos(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_acos(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
* atan
***********************************************/
array atan(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_atan(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
 * abs
 ***********************************************/
array abs(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_fabs(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
 * fabs
 ***********************************************/
array fabs(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_fabs(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
 * sqrt
 ***********************************************/
array sqrt(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_sqrt(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
 * rect
 ***********************************************/
array rect(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_rect(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
 * square
 ***********************************************/
array square(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_square(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
  * rand
  ***********************************************/
array rand(array& a)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_rand(c.dat,a.dat,a.n);
  c.n=a.n;
  return(c);
}


/***********************************************
 * conv cross convolution
 ***********************************************/
array conv(array& a,array& b)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_conv(c.dat,a.dat,a.n,b.dat,b.n);
  c.n=a.n;
  return(c);
}

/***********************************************
 * integ
 ***********************************************/
array integ(array& a,array& b)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_integ(c.dat,a.dat,b.dat,a.n);
  c.n=a.n;
  return(c);
}

/***********************************************
 * diff differential
 ***********************************************/
array diff(array& a,array& b)
{
  array c=array(0.0 , 0.0 , a.n);
  int i;
  G__ary_diff(c.dat,a.dat,b.dat,a.n);
  c.n=a.n;
  return(c);
}






/**********************************************************
 * subarray class for lvalue
 **********************************************************/

subarray array::sub(int offset,int ndat)
{
  subarray c;
  c.dat = dat + offset;
  if(offset+ndat>n) {
    c.n = n-offset;
    cerr << "Not enough data in master array, data shrinked\n";
  }
  else {
    c.n = ndat;
  }
  return(c);
}


array& subarray::operator =(array& a)
{
  if(a.n<n) memcpy(dat,a.dat,a.n*sizeof(double));
  else      memcpy(dat,a.dat,n*sizeof(double));
  return(a);
}

// optional for rvalue
subarray& subarray::operator =(subarray& a)
{
  if(a.n<n) memcpy(dat,a.dat,a.n*sizeof(double));
  else      memcpy(dat,a.dat,n*sizeof(double));
  return(a);
}


/***********************************************
 * subarray for rvalue
 ***********************************************/

array subarray(array& X,int offset,int ndat)
{
  array c=array(X,offset,ndat);
  return(c);
}



