/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/**************************************************************************
* DArray.h
*
* Array class template
*
**************************************************************************/

#ifndef DARRAY_H
#define DARRAY_H

#include <stdio.h>
#include <iostream>
#include <math.h>
#include <string.h>

#ifdef __GNUC__
extern int G__defaultsize;
#endif

class DArray;

DArray operator +(DArray& a,DArray& b);
DArray operator -(DArray& a,DArray& b);
DArray operator *(DArray& a,DArray& b);
DArray operator /(DArray& a,DArray& b);
DArray exp(DArray& a);
DArray abs(DArray& a);

class DArray  {
public:
  DArray(double start,double stop,int ndat);
  DArray(double x);
  DArray(DArray const & X);
  DArray(void);
  DArray(DArray& X,int offset,int ndat);
  ~DArray(); 
  
  DArray& operator =(DArray& a);
  DArray operator()(int from,int to);
  double& operator[](int index);
  int getsize(void) { return(n); }
  int resize(int size);
  static void setdefaultsize(int size) { G__defaultsize = size; }

  void disp(void);

  friend DArray operator +(DArray& a,DArray& b);
  friend DArray operator -(DArray& a,DArray& b);
  friend DArray operator *(DArray& a,DArray& b);
  friend DArray operator /(DArray& a,DArray& b);
  friend DArray exp(DArray& a);
  friend DArray abs(DArray& a);

#ifdef NOT_READY_YET
  friend DArray log(DArray& a);
  friend DArray log10(DArray& a);
  friend DArray sinc(DArray& a);
  friend DArray sin(DArray& a);
  friend DArray cos(DArray& a);
  friend DArray tan(DArray& a);
  friend DArray asin(DArray& a);
  friend DArray acos(DArray& a);
  friend DArray atan(DArray& a);
  friend DArray sinh(DArray& a);
  friend DArray cosh(DArray& a);
  friend DArray tanh(DArray& a);
  friend DArray sqrt(DArray& a);
  friend DArray rect(DArray& a);
  friend DArray square(DArray& a);
  friend DArray rand(DArray& a);
  friend DArray conv(DArray& a,DArray& b);
  friend DArray integ(DArray& x,DArray& y);
  friend DArray diff(DArray& x,DArray& y);
#endif

private:
  double *dat;         // pointer to data DArray
  int n;               // number of data
#ifndef __GNUC__
  static int G__defaultsize;
#endif
  int malloced;
  enum { ISOLD, ISNEW };
  DArray(double *p,int ndat,int isnew /* =0 */ );
} ;




#endif
