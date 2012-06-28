/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/**************************************************************************
* array.c
*
* Array class speed up library
*
*  makecint -dl array.sl -c array.c
*
*  Constructor, copy constructor, destructor, operator overloading
* function overloading, reference type
*
**************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#define G__ARRAYSL

void G__ary_assign(double *c,double start,double stop,int n)
{
  int i;
  double res;
  res = (stop-start)/(n-1);
  for(i=0;i<n;i++) c[i] = i*res + start ;
}

void G__ary_plus(double *c,double *a,double *b,int n)
{
  int i;
  for(i=0;i<n;i++) c[i] = a[i] + b[i];
}

void G__ary_minus(double *c,double *a,double *b,int n)
{
  int i;
  for(i=0;i<n;i++) c[i] = a[i] - b[i];
}

void G__ary_multiply(double *c,double *a,double *b,int n)
{
  int i;
  for(i=0;i<n;i++) c[i] = a[i] * b[i];
}

void G__ary_divide(double *c,double *a,double *b,int n)
{
  int i;
  for(i=0;i<n;i++) {
    if(b[i]!=0) c[i] = a[i] / b[i];
    else {
      if(a[i]==0) c[i]=1;
      else if(a[i]>0) c[i]=HUGE_VAL;
      else c[i] = -HUGE_VAL;
    }
  }
}

void G__ary_power(double *c,double *a,double *b,int n)
{
  int i,j;
  int flag=0;
  double r;
  for(i=0;i<n;i++) {
#ifndef G__OLDIMPLEMENTATION1622
    c[i] = pow(a[i],b[i]);
#else
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
#endif
  }
}

void G__ary_exp(double *c,double *a,int n)
{
  int i;
  for(i=0;i<n;i++) c[i] = exp(a[i]);
}

void G__ary_log(double *c,double *a,int n)
{
  int i;
  for(i=0;i<n;i++) {
    if(a[i]>0) c[i] = log(a[i]);
    else       c[i] = -HUGE_VAL;
  }
}

void G__ary_log10(double *c,double *a,int n)
{
  int i;
  for(i=0;i<n;i++) {
    if(a[i]>0) c[i] = log10(a[i]);
    else       c[i] = -HUGE_VAL;
  }
}

void G__ary_sinc(double *c,double *a,int n)
{
  int i;
  for(i=0;i<n;i++) {
    if(a[i]!=0) c[i] = sin(a[i])/a[i];
    else        c[i] = 1;
  }
}

void G__ary_sin(double *c,double *a,int n)
{
  int i;
  for(i=0;i<n;i++) c[i] = sin(a[i]);
}

void G__ary_cos(double *c,double *a,int n)
{
  int i;
  for(i=0;i<n;i++) c[i] = cos(a[i]);
}

void G__ary_tan(double *c,double *a,int n)
{
  int i;
  for(i=0;i<n;i++) c[i] = tan(a[i]);
}

void G__ary_sinh(double *c,double *a,int n)
{
  int i;
  for(i=0;i<n;i++) c[i] = sinh(a[i]);
}

void G__ary_cosh(double *c,double *a,int n)
{
  int i;
  for(i=0;i<n;i++) c[i] = cosh(a[i]);
}

void G__ary_tanh(double *c,double *a,int n)
{
  int i;
  for(i=0;i<n;i++) c[i] = tanh(a[i]);
}


void G__ary_asin(double *c,double *a,int n)
{
  int i;
  for(i=0;i<n;i++) c[i] = asin(a[i]);
}

void G__ary_acos(double *c,double *a,int n)
{
  int i;
  for(i=0;i<n;i++) c[i] = acos(a[i]);
}

void G__ary_atan(double *c,double *a,int n)
{
  int i;
  for(i=0;i<n;i++) c[i] = atan(a[i]);
}

void G__ary_fabs(double *c,double *a,int n)
{
  int i;
  for(i=0;i<n;i++) c[i] = fabs(a[i]);
}

void G__ary_sqrt(double *c,double *a,int n)
{
  int i;
  for(i=0;i<n;i++) c[i] = sqrt(a[i]);
}


void G__ary_rect(double *c,double *a,int n)
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

void G__ary_square(double *c,double *a,int n)
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

void G__ary_rand(double *c,double *a,int n)
{
  int i;
#if defined(__CINT__) || defined(G__WIN32)
  for(i=0;i<n;i++) c[i] = (double)rand()/0x3fffffff-1.0;
#else
  for(i=0;i<n;i++) c[i] = drand48();
#endif
}

void G__ary_conv(double *c,double *a,int n,double *b,int m)
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

void G__ary_integ(double *c,double *a,double *b,int n)
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

void G__ary_diff(double *c,double *a,double *b,int n)
{
  int i;
  double k,m;
  for(i=0;i<n-1;i++) {
    k = a[i+1]-a[i]; 
    if(k!=0) c[i] = (b[i+1]-b[i])/k;
    else {
      m = k*(b[i+1]-b[i]);
      if(m==0) {
	if(k>0) c[i] = 1;
	else    c[i] = -1;
      }
      else if(m>0) c[i] = HUGE_VAL;
      else         c[i] = -HUGE_VAL;
    }
  }
  c[i] = c[i-1];
}

void G__ary_max(double *c,double *a,double *b,int n)
{
  int i;
  for(i=0;i<n;i++) {
    if(a[i]<b[i]) c[i] = b[i];
    else          c[i] = a[i];
  }
}

void G__ary_min(double *c,double *a,double *b,int n)
{
  int i;
  for(i=0;i<n;i++) {
    if(a[i]>b[i]) c[i] = b[i];
    else          c[i] = a[i];
  }
}
