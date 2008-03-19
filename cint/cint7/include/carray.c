/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/**************************************************************************
* carray.c
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
#define G__CARRAYSL


void G__cary_multiply(double *cre,double *cim,double *are,double *aim,double *bre,double *bim,int n)
{
  int i;
  for(i=0;i<n;i++) {
    cre[i] = are[i]*bre[i] - aim[i]*bim[i];
    cim[i] = are[i]*bim[i] + aim[i]*bre[i];
  }
}

void G__cary_divide(double *cre,double *cim,double *are,double *aim,double *bre,double *bim,int n)
{
  int i;
  double x;
  for(i=0;i<n;i++) {
    x = bre[i]*bre[i]+bim[i]*bim[i];
    if(x!=0.0) {
      cre[i] = (are[i]*bre[i]+aim[i]*bim[i])/x;
      cim[i] = (aim[i]*bre[i]-are[i]*bim[i])/x;
    }
    else {
      cre[i]= 0;
      cim[i]= 0;
    }
  }
}


void G__cary_exp(double *cre,double *cim,double *are,double *aim,int n)
{
  int i;
  double mag;
  for(i=0;i<n;i++) {
    mag = exp(are[i]);
    cre[i] = mag*cos(aim[i]);
    cim[i] = mag*sin(aim[i]);
  }
}



void G__cary_fabs(double *c,double *are,double *aim,int n)
{
  int i;
  for(i=0;i<n;i++) {
    c[i] = sqrt(are[i]*are[i]+aim[i]*aim[i]);
  }
}

